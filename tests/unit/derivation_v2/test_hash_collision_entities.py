"""A session ``#X`` and a permanent ``X`` that share a bare name are separate entities."""
from __future__ import annotations

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.phase1_lineage import build_lineage_map
from app.derivation.v2.phase2_mutation_folder import fold_column_mutations

_SQL = """
CREATE PROCEDURE PRO.P AS
BEGIN
CREATE TABLE #AMH (CustomerAcID varchar(10), Status varchar(5))
INSERT INTO #AMH (CustomerAcID, Status)
SELECT A.CustomerAcID, A.SMA_CLASS FROM ##AccountCal A
INSERT INTO PRO.AMH (CustomerAcID, Status)
SELECT T.CustomerAcID, T.Status FROM #AMH T
END
"""

_NO_COLLISION_SQL = """
CREATE PROCEDURE PRO.P AS
BEGIN
CREATE TABLE #STAGE (CustomerAcID varchar(10), Status varchar(5))
INSERT INTO #STAGE (CustomerAcID, Status)
SELECT A.CustomerAcID, A.SMA_CLASS FROM ##AccountCal A
INSERT INTO PRO.FINAL (CustomerAcID, Status)
SELECT T.CustomerAcID, T.Status FROM #STAGE T
END
"""


def test_collision_is_detected_only_for_names_used_as_both_temp_and_permanent():
    assert build_lineage_map(_SQL, None).hash_collisions == {"AMH"}
    assert build_lineage_map(_NO_COLLISION_SQL, None).hash_collisions == set()


def test_temp_entity_sees_only_temp_writes_and_permanent_only_permanent_writes():
    lineage = build_lineage_map(_SQL, None)
    temp = fold_column_mutations(_SQL, "#AMH", "Status", lineage, None)
    perm = fold_column_mutations(_SQL, "AMH", "Status", lineage, None)
    assert len(temp) == 1 and "SMA_CLASS" in (temp[0].assigned_expression or "")
    assert len(perm) == 1
    # The permanent table reads the temp's column through the ``#``-qualified entity.
    assert "#AMH" in (perm[0].assigned_expression or "")


def test_hash_entity_is_quoted_with_its_hash():
    node = {"type": "COLUMN_REF", "entity": "#AMH", "column": "Status"}
    assert compile_ast_to_4x_string(node) == '"#AMH"."Status"'
    plain = {"type": "COLUMN_REF", "entity": "AMH", "column": "Status"}
    assert compile_ast_to_4x_string(plain) == '"AMH"."Status"'


def test_non_colliding_temps_keep_their_existing_names():
    lineage = build_lineage_map(_NO_COLLISION_SQL, None)
    temp = fold_column_mutations(_NO_COLLISION_SQL, "STAGE", "Status", lineage, None)
    assert len(temp) == 1 and "SMA_CLASS" in (temp[0].assigned_expression or "")
