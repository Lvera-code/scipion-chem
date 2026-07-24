# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors: Enzo Sierra (enzogael57@gmail.com)
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'you@yourinstitution.email'
# *
# **************************************************************************

"""
This protocol is used to predict B-cell epitope antigenicity with EpiDope
and extract epitope regions from its per-residue scores.

"""
import os
from pathlib import Path

import pandas as pd
from pyworkflow.object import Float
from pyworkflow.protocol import params
from pyworkflow.utils import Message
from pwem.protocols import EMProtocol

from pwchem import Plugin as pwchemPlugin
from pwchem.constants import EPIDOPE_DIC
from pwchem.objects import Sequence, SequenceROI, SetOfSequenceROIs

ACCESSION_COLUMN = "Accession"
RESIDUE_COLUMN = "Residue"
SCORE_COLUMN = "EpiDope score"

# Columns written by EpiDope in '<outdir>/epidope/<accession>.csv'
# (see epidope/epidope2.py::output_results in the EpiDope source).
_RAW_COLUMNS = ("position", "aminoacid", "score")


class EpidopeExecutionError(Exception):
    """EpiDope failed to run locally: missing installation, failed
    subprocess, timeout, or output in an unexpected format."""


class ProtEpiDopePrediction(EMProtocol):
    """
    AI Generated:

    Runs EpiDope (https://github.com/rnajena/EpiDope) locally to predict
    B-cell epitope antigenicity along a protein sequence, and extracts
    contiguous epitope regions from the resulting per-residue scores using a
    gap-tolerant sliding window.

    Overview
    --------
    1. Export the input sequence to a temporary FASTA file.
    2. Run the local 'epidope' binary (installed via this plugin's
       defineBinaries, see Plugin.addEpiDopePackage) against that FASTA.
    3. EpiDope writes one per-accession CSV under '<outdir>/epidope/', with
       one row per residue ('position', 'aminoacid', 'score').
    4. Slide a window (default 9 aa, the minimum B-cell recognition
       footprint) over the per-residue scores, tolerating a small number of
       individual residues below the threshold (max_gap_residues) so a real
       epitope is not discarded because of a single weak residue.
    5. Merge passing windows into contiguous regions and keep only those at
       least min_length residues long.

    Output
    ------
    outputROIs: SetOfSequenceROIs, one ROI per epitope region found, each
    annotated with its mean EpiDope score (_meanScore).
    """

    _label = 'epidope antigenicity prediction'

    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('inputSequence', params.PointerParam, pointerClass='Sequence',
                       label='Input protein sequence: ',
                       help='Protein sequence to run EpiDope antigenicity prediction on.')

        eGroup = form.addGroup('Epitope extraction')
        eGroup.addParam('threshold', params.FloatParam, label='Score threshold: ', default=0.818,
                         help='EpiDope score threshold above which a residue/window is considered part '
                              'of a candidate epitope.')
        eGroup.addParam('minLength', params.IntParam, label='Minimum epitope length: ', default=9,
                         help='Minimum length (aa) of a merged region to be reported as an epitope.')
        eGroup.addParam('windowSize', params.IntParam, label='Sliding window size: ', default=9,
                         expertLevel=params.LEVEL_ADVANCED,
                         help='Minimum B-cell recognition footprint (9 aa by default).')
        eGroup.addParam('maxGapResidues', params.IntParam, label='Max gap residues per window: ', default=2,
                         expertLevel=params.LEVEL_ADVANCED,
                         help='Individual below-threshold residues tolerated within a single window, so '
                              'a real epitope is not discarded because of a single weak residue.')

    def _insertAllSteps(self):
        self._insertFunctionStep(self.epidopeStep)
        self._insertFunctionStep(self.createOutputStep)

    # ---------------------------------- Steps -----------------------------------

    def writeInputFasta(self):
        faFile = self._getExtraPath('inputSequence.fa')
        self.inputSequence.get().exportToFile(faFile)
        return os.path.abspath(faFile)

    def epidopeStep(self):
        epidope_bin = pwchemPlugin.getProgramHome(EPIDOPE_DIC, os.path.join('bin', 'epidope'))

        fasta_file = self.writeInputFasta()
        out_dir = os.path.abspath(self._getExtraPath('epidope_raw'))
        os.makedirs(out_dir, exist_ok=True)

        args = f'-i {fasta_file} -o {out_dir} -t {self.threshold.get()}'
        # EpiDope's 'epidope' binary is a self-contained shim (shebang points
        # to its own conda env's python): invoke it directly, do NOT go
        # through 'conda run' or an activation command (those produced
        # intermittent spurious failures with piped stdout/stderr in testing).
        self.runJob(epidope_bin, args)

    def createOutputStep(self):
        out_dir = self._getExtraPath('epidope_raw')
        raw_df = self._loadRawScores(out_dir)

        epitopes_df = self._extractEpitopeRegions(
            raw_df,
            threshold=self.threshold.get(),
            min_length=self.minLength.get(),
            window_size=self.windowSize.get(),
            max_gap_residues=self.maxGapResidues.get(),
        )

        inputSeq = self.inputSequence.get()
        outROIs = SetOfSequenceROIs(filename=self._getPath('sequenceROIs.sqlite'))
        for row in epitopes_df.itertuples(index=False):
            roiId = f'ROI_{row.start}-{row.end}'
            roiSeq = Sequence(sequence=row.sequence, name=roiId, id=roiId,
                               description='EpiDope epitope')
            seqROI = SequenceROI(sequence=inputSeq, seqROI=roiSeq, roiIdx=row.start, roiIdx2=row.end)
            seqROI._meanScore = Float(row.mean_score)
            outROIs.append(seqROI)

        if len(outROIs) > 0:
            self._defineOutputs(outputROIs=outROIs)
            self._defineSourceRelation(self.inputSequence, outROIs)

    # ---------------------------------- Utils -----------------------------------

    @staticmethod
    def _loadRawScores(out_dir: str) -> pd.DataFrame:
        """Concatenate the per-accession CSVs EpiDope writes in '<out_dir>/epidope/'."""
        per_gene_dir = Path(out_dir) / 'epidope'
        csv_files = sorted(per_gene_dir.glob('*.csv'))
        if not csv_files:
            found = sorted(p.name for p in Path(out_dir).rglob('*') if p.is_file())
            raise EpidopeExecutionError(
                f"No per-accession score CSV found in '{per_gene_dir}'. "
                f"Files generated by EpiDope: {found or '<none>'}."
            )

        frames = []
        for csv_path in csv_files:
            df = pd.read_csv(csv_path, sep='\t')
            missing = set(_RAW_COLUMNS) - set(df.columns)
            if missing:
                raise EpidopeExecutionError(
                    f"Output CSV '{csv_path}' is missing expected columns "
                    f"{sorted(missing)}. Columns found: {list(df.columns)}."
                )
            df = df.sort_values('position').reset_index(drop=True)
            df.insert(0, ACCESSION_COLUMN, csv_path.stem)
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.rename(columns={'aminoacid': RESIDUE_COLUMN, 'score': SCORE_COLUMN})
        return combined[[ACCESSION_COLUMN, RESIDUE_COLUMN, SCORE_COLUMN]]

    @staticmethod
    def _extractEpitopeRegions(raw_df: pd.DataFrame, threshold: float, min_length: int,
                                window_size: int, max_gap_residues: int) -> pd.DataFrame:
        """Slide a window over per-residue scores and merge passing windows into
        contiguous epitope regions, tolerating up to max_gap_residues
        below-threshold residues per window."""
        records = []
        for accession, group in raw_df.groupby(ACCESSION_COLUMN, sort=False):
            residues = group[RESIDUE_COLUMN].tolist()
            scores = group[SCORE_COLUMN].tolist()
            n = len(residues)

            passing = [False] * n
            for i in range(n - window_size + 1):
                windowScores = scores[i:i + window_size]
                nBelow = sum(1 for s in windowScores if s < threshold)
                if nBelow <= max_gap_residues:
                    for j in range(i, i + window_size):
                        passing[j] = True

            start = None
            for i in range(n + 1):
                if i < n and passing[i]:
                    if start is None:
                        start = i
                else:
                    if start is not None:
                        length = i - start
                        if length >= min_length:
                            records.append({
                                'start': start + 1,
                                'end': i,
                                'sequence': ''.join(residues[start:i]),
                                'mean_score': sum(scores[start:i]) / length,
                            })
                        start = None

        return pd.DataFrame.from_records(
            records, columns=['start', 'end', 'sequence', 'mean_score']
        )

    # ---------------------------------- Info -------------------------------

    def _summary(self):
        summary = []
        if self.isFinished():
            outROIs = getattr(self, 'outputROIs', None)
            n = len(outROIs) if outROIs is not None else 0
            summary.append(f'{n} epitope region(s) found above threshold {self.threshold.get()}.')
        return summary
