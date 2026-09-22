-- ============================================================
-- Fixture for Derivation Engine v2 verification
-- Mirrors PRO.Final_AssetClass_Npadate_MOC patterns:
--   ##AccountCal as root interface entity
--   local # temp staging with SELECT INTO
--   sequential UPDATE passes on FinalAssetClassAlt_Key
-- ============================================================
CREATE PROCEDURE PRO.Final_AssetClass_Npadate_MOC
    @TIMEKEY INT
AS
BEGIN
    SET NOCOUNT ON

    IF OBJECT_ID('tempdb..#AssetClassWork') IS NOT NULL DROP TABLE #AssetClassWork

    SELECT
        A.AccountEntityID,
        A.CustomerEntityID,
        A.AssetClassAlt_Key,
        A.FinalAssetClassAlt_Key,
        A.NPA_Date,
        C.CustSegment
    INTO #AssetClassWork
    FROM ##AccountCal A
    INNER JOIN ##CUSTOMERCAL C
        ON A.CustomerEntityID = C.CustomerEntityID
    WHERE A.EffectiveFromTimeKey <= @TIMEKEY
      AND A.EffectiveToTimeKey >= @TIMEKEY

    -- Pass 1: default FinalAssetClass from current asset class
    UPDATE W
    SET W.FinalAssetClassAlt_Key = W.AssetClassAlt_Key
    FROM #AssetClassWork W

    -- Pass 2: force STANDARD when NPA date is empty
    UPDATE W
    SET W.FinalAssetClassAlt_Key = 1
    FROM #AssetClassWork W
    WHERE W.NPA_Date IS NULL

    -- Pass 3: override from customer segment for SMA / watchlist
    UPDATE W
    SET W.FinalAssetClassAlt_Key = 2
    FROM #AssetClassWork W
    INNER JOIN ##CUSTOMERCAL C
        ON W.CustomerEntityID = C.CustomerEntityID
    WHERE C.CustSegment IN ('SMA1', 'SMA2')
       OR W.AssetClassAlt_Key IN (2, 3)

    -- Pass 4: write folded result back to root interface entity
    UPDATE A
    SET A.FinalAssetClassAlt_Key = W.FinalAssetClassAlt_Key
    FROM ##AccountCal A
    INNER JOIN #AssetClassWork W
        ON A.AccountEntityID = W.AccountEntityID

    IF @TIMEKEY > 26267
    BEGIN
        UPDATE A
        SET A.FinalAssetClassAlt_Key = 4
        FROM ##AccountCal A
        WHERE A.FinalAssetClassAlt_Key IS NULL
          AND A.AssetClassAlt_Key = 4
    END
END
GO
