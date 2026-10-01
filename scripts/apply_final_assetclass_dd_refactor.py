"""Apply DD-friendly refactor to PRO.Final_AssetClass_Npadate (behavior-preserving)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_REPO = ROOT / "samples" / "sql" / "PRO.Final_AssetClass_Npadate.StoredProcedure.sql"
_CANDIDATES = [
    ROOT / "samples" / "sql" / "PRO.Final_AssetClass_Npadate.StoredProcedure.sql",
    Path(r"C:\Users\dishaj\Downloads\PRO_SPs\PRO_SPs\PRO.Final_AssetClass_Npadate.StoredProcedure.sql"),
]
SYNC_SNIP = (ROOT / "scripts" / "sql_snippets" / "final_assetclass_customer_sync.sql").read_text(encoding="utf-8")

HEADER = """/*=================================================================================
  AUTHER      : TRILOKI KHANNA
  CREATE DATE : 27-11-2019
  MODIFY DATE : 27-11-2019
  DESCRIPTION : UPDATE FINAL ASSET CLASS AND MIN NPA DATE UPDATE CUSTOMER LEVEL AT ACCOUNT LEVEL
  EXEC [PRO].[Final_AssetClass_Npadate] 25233

  REFACTORED for SQL-to-DD Report Generator compatibility.
  Structural blockers addressed:
    1. NPA_Reason dynamic LIKE -> #NPA_Reason_Match.Stg_NpaReason_MatchFlag
    2. STRING_AGG rollup -> #Data / #NPADegReason.Stg_Aggregated_DegReason staging chain
    3. FinalNpaDt / DegReason chronology -> #Temp_NPA_Calc checkpoints (lineage)
    4. Duplicate customer->account sync UPDATEs -> consolidated 3-arm ROW_NUMBER sync
===================================================================================*/"""

TEMP_AND_LINEAGE = """
		IF OBJECT_ID('TEMPDB..#Temp_NPA_Calc') IS NOT NULL
			DROP TABLE #Temp_NPA_Calc

		CREATE TABLE #Temp_NPA_Calc
		(
			StepOrder             INT,
			StepName              VARCHAR(100),
			AccountEntityID       BIGINT NULL,
			CustomerEntityID      BIGINT NULL,
			UcifEntityID          VARCHAR(100) NULL,
			FinalAssetClassAlt_Key INT NULL,
			FinalNpaDt            DATE NULL,
			DegReason             VARCHAR(1000) NULL,
			CapturedAt            DATETIME NOT NULL DEFAULT GETDATE()
		)

		IF OBJECT_ID('TEMPDB..#Lineage_Log') IS NOT NULL
			DROP TABLE #Lineage_Log

		CREATE TABLE #Lineage_Log
		(
			StepOrder     INT,
			StepName      VARCHAR(100),
			EntityName    VARCHAR(100),
			ColumnName    VARCHAR(100),
			KeyValue      VARCHAR(100) NULL,
			CapturedValue VARCHAR(500) NULL,
			CapturedAt    DATETIME NOT NULL DEFAULT GETDATE()
		)

		INSERT INTO #Temp_NPA_Calc (StepOrder, StepName, AccountEntityID, CustomerEntityID, UcifEntityID, FinalAssetClassAlt_Key, FinalNpaDt, DegReason)
		SELECT 10, 'Step10_CustomerToAccount_NpaSync', AccountEntityID, CustomerEntityID, UcifEntityID, FinalAssetClassAlt_Key, FinalNpaDt, DegReason
		FROM ##ACCOUNTCAL
"""

NPA_REASON_BLOCK = """
		IF OBJECT_ID('TEMPDB..#NPA_Reason_Match') IS NOT NULL
			DROP TABLE #NPA_Reason_Match

		SELECT
			A.CustomerAcID,
			CASE WHEN A.NPA_Reason IS NOT NULL
					AND A.NPA_Reason <> ''
					AND CHARINDEX(A.NPA_Reason, B.DegradeReason) > 0
				THEN 1 ELSE 0 END AS Stg_NpaReason_MatchFlag
		INTO #NPA_Reason_Match
		FROM ##ACCOUNTCAL A
		INNER JOIN (
			SELECT DISTINCT CUSTOMERACID, PERC_FinalAssetClass_AltKey, PERC_FinalNpaDt, DegradeReason
			FROM PRO.CoBorrowerCal
		) B ON A.CustomerAcID = B.CustomerACID
		WHERE (A.Asset_Norm NOT IN ('AlWYS_STD') OR A.MOCTYPE IS NULL)
			AND B.PERC_FinalAssetClass_AltKey > 1

		UPDATE A
		SET
			A.FinalAssetClassAlt_Key = B.PERC_FinalAssetClass_AltKey,
			A.FinalNpaDt = B.PERC_FinalNpaDt,
			A.NPA_Reason = CASE WHEN M.Stg_NpaReason_MatchFlag = 1
				THEN A.NPA_Reason
				ELSE CONCAT(A.NPA_Reason, ',', B.DegradeReason) END,
			A.DegReason = B.DegradeReason
		FROM ##ACCOUNTCAL A
		INNER JOIN (
			SELECT DISTINCT CUSTOMERACID, PERC_FinalAssetClass_AltKey, PERC_FinalNpaDt, DegradeReason
			FROM PRO.CoBorrowerCal
		) B ON A.CustomerAcID = B.CustomerACID
		INNER JOIN #NPA_Reason_Match M ON A.CustomerAcID = M.CustomerAcID
		WHERE (A.Asset_Norm NOT IN ('AlWYS_STD') OR A.MOCTYPE IS NULL)
			AND B.PERC_FinalAssetClass_AltKey > 1

		INSERT INTO #Temp_NPA_Calc (StepOrder, StepName, AccountEntityID, FinalAssetClassAlt_Key, FinalNpaDt, DegReason)
		SELECT 32, 'Step32_CoBorrower_Final_NpaReason_DegReason', AccountEntityID, FinalAssetClassAlt_Key, FinalNpaDt, DegReason
		FROM ##ACCOUNTCAL
		WHERE (Asset_Norm NOT IN ('AlWYS_STD') OR MOCTYPE IS NULL)
"""

OLD_TRIPLE_SYNC = """	 UPDATE A SET 
	         A.FinalAssetClassAlt_Key=ISNULL(B.SysAssetClassAlt_Key,1)
		    ,A.FinalNpaDt=B.SysNPA_Dt
			FROM ##ACCOUNTCAL A INNER   JOIN ##CustomerCal B 
			ON  A.RefCustomerID=B.RefCustomerID AND A.SourceSystemCustomerID=B.SourceSystemCustomerID 
			WHERE ISNULL(B.SysAssetClassAlt_Key,1)<>1 AND B.RefCustomerID<>'0'

	UPDATE A SET 
	         A.FinalAssetClassAlt_Key=ISNULL(B.SysAssetClassAlt_Key,1)
		    ,A.FinalNpaDt=B.SysNPA_Dt
			FROM ##ACCOUNTCAL A INNER   JOIN ##CustomerCal B 
			ON  A.SourceSystemCustomerID=B.SourceSystemCustomerID 
			WHERE ISNULL(B.SysAssetClassAlt_Key,1)<>1

	UPDATE A SET 
	         A.FinalAssetClassAlt_Key=ISNULL(B.SysAssetClassAlt_Key,1)
		    ,A.FinalNpaDt=B.SysNPA_Dt
			FROM ##ACCOUNTCAL A INNER   JOIN ##CustomerCal B 
			ON  A.UcifEntityID=B.UcifEntityID 
			WHERE ISNULL(B.SysAssetClassAlt_Key,1)<>1"""


def main() -> None:
    src = next((p for p in _CANDIDATES if p.is_file()), None)
    if src is None:
        raise SystemExit(f"No source SP found; expected one of: {_CANDIDATES}")
    text = src.read_text(encoding="utf-8")

    text = text.replace(
        """/*=========================================
 AUTHER : TRILOKI KHANNA
 CREATE DATE : 27-11-2019
 MODIFY DATE : 27-11-2019
 DESCRIPTION :UPDATE FINAL ASSET CLASS AND MIN NPA DATE UPDATE CUSTOMER LEVEL AT ACCOUNT LEVEL
 EXEC [PRO].[Final_AssetClass_Npadate] 25233
=============================================*/""",
        HEADER,
    )

    marker = "AND ISNULL(A.FlgDeg,'N')='Y'\n\n\nUPDATE A SET A.FINALASSETCLASSALT_KEY"
    if marker in text:
        text = text.replace(
            marker,
            "AND ISNULL(A.FlgDeg,'N')='Y'\n" + TEMP_AND_LINEAGE + "\n\nUPDATE A SET A.FINALASSETCLASSALT_KEY",
        )

    old_npa = """UPDATE 
A 
SET 
A.FinalAssetClassAlt_Key=B.PERC_FinalAssetClass_AltKey,
A.FinalNpaDt=B.PERC_FinalNpaDt,
A.NPA_Reason=(CASE WHEN B.DegradeReason LIKE '%' + A.NPA_Reason + '%'
                   THEN A.NPA_Reason
                   ELSE CONCAT(A.NPA_Reason, ',', b.DegradeReason) END),
A.DegReason=b.DegradeReason
FROM 
##ACCOUNTCAL A
INNER JOIN 
(SELECT DISTINCT CUSTOMERACID,PERC_FinalAssetClass_AltKey,PERC_FinalNpaDt,DegradeReason
FROM PRO.CoBorrowerCal) B
ON A.CustomerAcID=B.CustomerACID

--WHERE A.Asset_Norm NOT IN('AlWYS_STD')
WHERE (A.Asset_Norm NOT IN ('AlWYS_STD') OR A.MOCTYPE IS NULL) ----------Added by Prashant 20022025 as discussion with Akshay sir , Kandpal sir and Jayadev ----
AND B.PERC_FinalAssetClass_AltKey>1"""

    if old_npa not in text:
        raise SystemExit("NPA_Reason block not found")
    text = text.replace(old_npa, NPA_REASON_BLOCK.strip())

    if text.count(OLD_TRIPLE_SYNC) != 2:
        raise SystemExit(f"Expected 2 triple-sync blocks, found {text.count(OLD_TRIPLE_SYNC)}")
    text = text.replace(OLD_TRIPLE_SYNC, SYNC_SNIP.strip())

    text = text.replace(
        "select UcifEntityID, STRING_AGG(DegReason, ', ') DegReason\n\tinto  #NPADegReason from #Data",
        "SELECT UcifEntityID, STRING_AGG(DegReason, ', ') AS Stg_Aggregated_DegReason\n\tINTO #NPADegReason FROM #Data",
    )
    text = text.replace(
        "UPDATE A SET DegReason=B.DegReason  FROM ##CustomerCal A \n\tINNER JOIN #NPADegReason B  ON A.UcifEntityID=B.UcifEntityID",
        "UPDATE A SET DegReason=B.Stg_Aggregated_DegReason FROM ##CustomerCal A\n\tINNER JOIN #NPADegReason B ON A.UcifEntityID=B.UcifEntityID",
    )

    final_sync_marker = SYNC_SNIP.strip() + "\n\t\t\n\t "
    # After second consolidated sync (buyout section), add checkpoint 51
    buyout_end = "/* END OF LOAN BUYOUT */"
    idx = text.rfind(buyout_end)
    if idx == -1:
        raise SystemExit("buyout end marker missing")
    insert_51 = (
        "\n\n\t\tINSERT INTO #Temp_NPA_Calc (StepOrder, StepName, AccountEntityID, FinalAssetClassAlt_Key, FinalNpaDt, DegReason)\n"
        "\t\tSELECT 51, 'Step51_FinalAssetClass_NpaDt_Sync', AccountEntityID, FinalAssetClassAlt_Key, FinalNpaDt, DegReason\n"
        "\t\tFROM ##ACCOUNTCAL\n"
    )
    # Insert after last occurrence of sync block before degreason copy
    pos = text.rfind(SYNC_SNIP.strip())
    if pos == -1:
        raise SystemExit("sync block missing after replace")
    end_sync = pos + len(SYNC_SNIP.strip())
    text = text[:end_sync] + insert_51 + text[end_sync:]

    always_std = "UPDATE A SET FinalAssetClassAlt_Key=1,FinalNpaDt=NULL, DEGREASON=NULL FROM ##ACCOUNTCAL A WHERE A.ASSET_NORM ='ALWYS_STD'"
    if always_std in text:
        text = text.replace(
            always_std,
            always_std
            + "\n\n\t\tINSERT INTO #Temp_NPA_Calc (StepOrder, StepName, AccountEntityID, FinalAssetClassAlt_Key, FinalNpaDt, DegReason)\n"
            "\t\tSELECT 60, 'Step60_Final_AlwaysStd_Reset', AccountEntityID, FinalAssetClassAlt_Key, FinalNpaDt, DegReason\n"
            "\t\tFROM ##ACCOUNTCAL WHERE ASSET_NORM = 'ALWYS_STD'",
        )

    agg_update = "AND A.FlgDeg='Y'\n\n\tUpdate ##CustomerCal"
    if agg_update in text:
        text = text.replace(
            agg_update,
            "AND A.FlgDeg='Y'\n\n\t\tINSERT INTO #Temp_NPA_Calc (StepOrder, StepName, CustomerEntityID, DegReason)\n"
            "\t\tSELECT 70, 'Step70_CustomerCal_DegReason_FromAgg', RefCustomerID, DegReason\n"
            "\t\tFROM ##CustomerCal WHERE FlgDeg='Y'\n\n\tUpdate ##CustomerCal",
        )

    # Pass1 buyout lineage (after pre-erosion buyout update)
    pass1_lineage = """\t\tUPDATE B
\t\t\tSET B.NPA_FLAG='NPA_OTHERS'
\t\tFROM ##ACCOUNTCAL A
\t\t\tINNER JOIN #CTE_BB_New AA
\t\t\t\tON A.UcifEntityID=A.UcifEntityID
\t\t\tINNER JOIN PRO.BuyoutUploadDetailsCal  B
\t\t\t\tON B.AccountEntityId=A.AccountEntityId


/* END OF PERCOLATION WORK */"""

    pass1_lineage_fixed = """\t\tUPDATE B
\t\t\tSET B.NPA_FLAG='NPA_OTHERS'
\t\tFROM ##ACCOUNTCAL A
\t\t\tINNER JOIN #CTE_BB_New AA
\t\t\t\tON A.UcifEntityID=A.UcifEntityID
\t\t\tINNER JOIN PRO.BuyoutUploadDetailsCal  B
\t\t\t\tON B.AccountEntityId=A.AccountEntityId

\t\tINSERT INTO #Lineage_Log (StepOrder, StepName, EntityName, ColumnName, KeyValue, CapturedValue)
\t\tSELECT 23, 'Step23_Pass1_BuyoutNpaFlag', 'BuyoutUploadDetailsCal', 'NPA_FLAG',
\t\t\tCAST(B.AccountEntityId AS VARCHAR(50)), B.NPA_FLAG
\t\tFROM PRO.BuyoutUploadDetailsCal B
\t\t\tINNER JOIN ##ACCOUNTCAL A ON B.AccountEntityId = A.AccountEntityId
\t\t\tINNER JOIN #CTE_BB_New AA ON A.UcifEntityID = AA.UcifEntityID

/* END OF PERCOLATION WORK */"""

    if pass1_lineage in text:
        text = text.replace(pass1_lineage, pass1_lineage_fixed)

    buyout_pass2 = """\t\tUPDATE B
\t\t\tSET B.NPA_FLAG='NPA_OTHERS'
\t\tFROM ##ACCOUNTCAL A
\t\t\tINNER JOIN #CTE_BB AA
\t\t\t\tON A.UcifEntityID=A.UcifEntityID
\t\t\tINNER JOIN PRO.BuyoutUploadDetailsCal  B
\t\t\t\tON B.AccountEntityId=A.AccountEntityId


\t
\t\t/* END OF NPA FLAG UPDATE*/"""

    buyout_pass2_fixed = """\t\tUPDATE B
\t\t\tSET B.NPA_FLAG='NPA_OTHERS'
\t\tFROM ##ACCOUNTCAL A
\t\t\tINNER JOIN #CTE_BB AA
\t\t\t\tON A.UcifEntityID=A.UcifEntityID
\t\t\tINNER JOIN PRO.BuyoutUploadDetailsCal  B
\t\t\t\tON B.AccountEntityId=A.AccountEntityId

\t\tINSERT INTO #Lineage_Log (StepOrder, StepName, EntityName, ColumnName, KeyValue, CapturedValue)
\t\tSELECT 50, 'Step50_Pass2_BuyoutNpaFlag', 'BuyoutUploadDetailsCal', 'NPA_FLAG',
\t\t\tCAST(B.AccountEntityId AS VARCHAR(50)), B.NPA_FLAG
\t\tFROM PRO.BuyoutUploadDetailsCal B
\t\t\tINNER JOIN ##ACCOUNTCAL A ON B.AccountEntityId = A.AccountEntityId
\t\t\tINNER JOIN #CTE_BB AA ON A.UcifEntityID = AA.UcifEntityID

\t
\t\t/* END OF NPA FLAG UPDATE*/"""

    if buyout_pass2 in text:
        text = text.replace(buyout_pass2, buyout_pass2_fixed)

    OUT_REPO.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPO.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPO.write_text(text, encoding="utf-8")
    pro_sp = _CANDIDATES[1]
    if pro_sp.is_file() or pro_sp.parent.exists():
        pro_sp.write_text(text, encoding="utf-8")
    print(f"Wrote {OUT_REPO}")
    if pro_sp.is_file():
        print(f"Wrote {pro_sp}")


if __name__ == "__main__":
    main()
