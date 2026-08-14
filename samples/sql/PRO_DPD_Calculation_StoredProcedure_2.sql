-- ============================================================
-- Oracle 19c PL/SQL Conversion
-- Source : PRO.DPD_Calculation (SQL Server)
-- Target : PRO.DPD_Calculation (Oracle 19c)
-- Rules  : No GTT / No ORA$PTT_
--
-- IMPORTANT: Source uses ##AccountCal (SQL Server global temp table)
-- pre-populated by the calling batch framework before this proc is called.
-- Oracle equivalent: PRO.AccountCal_Stg (permanent staging table).
-- The calling framework must TRUNCATE and populate PRO.AccountCal_Stg
-- before calling this procedure.
--
-- Changes:
--   ##AccountCal         → PRO.AccountCal_Stg (permanent staging table)
--   DATEDIFF(DAY,d1,d2)  → (d2 - d1)
--   DATE literals         → DATE 'YYYY-MM-DD'
--   ISNULL               → NVL
--   GETDATE()            → SYSDATE
--   UPDATE...FROM JOIN   → MERGE INTO...USING
--   @var                 → v_var / p_var
--   SET NOCOUNT ON/OFF   → removed
--   WITH RECOMPILE       → removed
--   BEGIN TRY/CATCH      → BEGIN...EXCEPTION WHEN OTHERS
--   SysDayMatrix.Date    → SysDayMatrix."Date" (Oracle reserved word)
--   COUNT column         → "COUNT" (Oracle function name)
--   IF OBJECT_ID(TEMPDB..) → removed (no temp table)
--   IF @TIMEKEY > 26267  → IF p_TIMEKEY > 26267 THEN
--   IF @TIMEKEY > 26384  → nested CASE WHEN p_TIMEKEY > 26384
-- ============================================================
CREATE OR REPLACE PROCEDURE PRO.DPD_Calculation(
    p_TIMEKEY IN NUMBER
)
AS
    v_ProcessDate DATE;
BEGIN
    BEGIN
        SELECT "Date" INTO v_ProcessDate
        FROM SysDayMatrix
        WHERE TimeKey = p_TIMEKEY AND ROWNUM = 1;
    EXCEPTION WHEN NO_DATA_FOUND THEN v_ProcessDate := NULL;
    END;

    BEGIN

        UPDATE PRO.AccountCal_Stg SET IntNotServicedDt = NULL
        WHERE (IntNotServicedDt = DATE '1900-01-01' OR IntNotServicedDt = TO_DATE('01/01/1900','DD/MM/YYYY'));

        UPDATE PRO.AccountCal_Stg SET LastCrDate = NULL
        WHERE (LastCrDate = DATE '1900-01-01' OR LastCrDate = TO_DATE('01/01/1900','DD/MM/YYYY'));

        UPDATE PRO.AccountCal_Stg SET ContiExcessDt = NULL
        WHERE (ContiExcessDt = DATE '1900-01-01' OR ContiExcessDt = TO_DATE('01/01/1900','DD/MM/YYYY'));

        UPDATE PRO.AccountCal_Stg SET OverDueSinceDt = NULL
        WHERE (OverDueSinceDt = DATE '1900-01-01' OR OverDueSinceDt = TO_DATE('01/01/1900','DD/MM/YYYY'));

        UPDATE PRO.AccountCal_Stg SET ReviewDueDt = NULL
        WHERE (ReviewDueDt = DATE '1900-01-01' OR ReviewDueDt = TO_DATE('01/01/1900','DD/MM/YYYY'));

        UPDATE PRO.AccountCal_Stg SET StockStDt = NULL
        WHERE (StockStDt = DATE '1900-01-01' OR StockStDt = TO_DATE('01/01/1900','DD/MM/YYYY'));

        /*------------------INITIAL ALL DPD 0 FOR RE-PROCESSING------------------------------- */
        UPDATE PRO.AccountCal_Stg A
        SET A.DPD_IntService      = 0,
            A.DPD_NoCredit        = 0,
            A.DPD_Overdrawn       = 0,
            A.DPD_Overdue         = 0,
            A.DPD_Renewal         = 0,
            A.DPD_StockStmt       = 0,
            DPD_PrincOverdue      = 0,
            DPD_IntOverdueSince   = 0,
            DPD_OtherOverdueSince = 0;

        /*---------- CALCULATED ALL DPD---------------------------------------------------------*/
        IF p_TIMEKEY > 26267 THEN  ----IMPLEMENTED FROM 2021-12-01
            UPDATE PRO.AccountCal_Stg A
            SET A.DPD_IntService  = CASE WHEN p_TIMEKEY > 26384  --28032022 amar - implemented 90 days summation ccod credit and intt logic
                                         THEN (CASE WHEN A.IntNotServicedDt IS NOT NULL THEN (v_ProcessDate - A.IntNotServicedDt) + 2 ELSE 0 END)
                                         ELSE (CASE WHEN A.IntNotServicedDt IS NOT NULL THEN (v_ProcessDate - A.IntNotServicedDt) + 1 ELSE 0 END)
                                    END,
---             ,A.DPD_NoCredit =  (CASE WHEN A.LastCrDate IS NOT NULL THEN DATEDIFF(DAY,A.LastCrDate,@ProcessDate) ELSE 0 END)
                --,A.DPD_NoCredit = CASE WHEN (DebitSinceDt IS NULL OR DATEDIFF(DAY,DebitSinceDt,@ProcessDate)>90)
                A.DPD_NoCredit    = CASE WHEN (DebitSinceDt IS NULL OR (v_ProcessDate - DebitSinceDt) >= 90)
                -------ABOVE CHANGES DONE BY PRASHANT ON 30032026 AS DISCUSSED WITH ANIRUDDH SIR---
                                         THEN (CASE WHEN A.LastCrDate IS NOT NULL THEN (v_ProcessDate - A.LastCrDate) + 1 ELSE 0 END)
                                         ELSE 0 END,
                A.DPD_Overdrawn   = (CASE WHEN A.ContiExcessDt IS NOT NULL THEN (v_ProcessDate - A.ContiExcessDt) + 1 ELSE 0 END),
                A.DPD_Overdue     = CASE WHEN p_TIMEKEY > 26372
                                        ------ AMAR - CHANGES ON 17032021 AS PER EMAIL BY ASHISH SIR DATED - 17-03-2021 1:59 PM - SUBJECT - Credit Card NPA Computation
                                        THEN (CASE WHEN A.OverDueSinceDt IS NOT NULL THEN (v_ProcessDate - A.OverDueSinceDt) + 1 ELSE 0 END)
                                        ELSE (CASE WHEN A.OverDueSinceDt IS NOT NULL THEN (v_ProcessDate - A.OverDueSinceDt) + (CASE WHEN SourceAlt_Key = 6 THEN 0 ELSE 1 END) ELSE 0 END)
                                    END,
                A.DPD_Renewal     = (CASE WHEN A.ReviewDueDt IS NOT NULL THEN (v_ProcessDate - A.ReviewDueDt) + 1 ELSE 0 END),
                A.DPD_StockStmt   = (CASE WHEN A.StockStDt IS NOT NULL THEN (v_ProcessDate - A.StockStDt) + 1 ELSE 0 END);
        ELSE
            UPDATE PRO.AccountCal_Stg A
            SET A.DPD_IntService  = (CASE WHEN A.IntNotServicedDt IS NOT NULL THEN (v_ProcessDate - A.IntNotServicedDt) ELSE 0 END),
---             ,A.DPD_NoCredit =  (CASE WHEN A.LastCrDate IS NOT NULL THEN DATEDIFF(DAY,A.LastCrDate,@ProcessDate) ELSE 0 END)
                A.DPD_NoCredit    = CASE WHEN (DebitSinceDt IS NULL OR (v_ProcessDate - DebitSinceDt) > 90)
                                         THEN (CASE WHEN A.LastCrDate IS NOT NULL THEN (v_ProcessDate - A.LastCrDate) ELSE 0 END)
                                         ELSE 0 END,
                A.DPD_Overdrawn   = (CASE WHEN A.ContiExcessDt IS NOT NULL THEN (v_ProcessDate - A.ContiExcessDt) + 1 ELSE 0 END),
                A.DPD_Overdue     = (CASE WHEN A.OverDueSinceDt IS NOT NULL THEN (v_ProcessDate - A.OverDueSinceDt) ELSE 0 END),
                A.DPD_Renewal     = (CASE WHEN A.ReviewDueDt IS NOT NULL THEN (v_ProcessDate - A.ReviewDueDt) ELSE 0 END),
                A.DPD_StockStmt   = (CASE WHEN A.StockStDt IS NOT NULL THEN (v_ProcessDate - A.StockStDt) ELSE 0 END);
        END IF;

        /* AMAR --DEBIT SINCE DATE DPD CALCULATION AND UPDATE IN DPD_Overdrawn AS DISCUSSED WITH SHARMA SIR AND TRILOKI SIR ON 31082021 */
        /*   amar --commented as per call by Ashish Sir on 02092021
        UPDATE A SET A.DPD_Overdrawn=(CASE WHEN A.DebitSinceDt IS NOT NULL THEN DATEDIFF(DAY,A.DebitSinceDt,@ProcessDate)+1 ELSE 0 END)
        FROM PRO.AccountCal A INNER JOIN DimProduct B ON (B.EffectiveFromTimeKey<=@TIMEKEY AND B.EffectiveToTimeKey>=@TIMEKEY) AND A.ProductAlt_Key=B.ProductAlt_Key
        WHERE B.SchemeType='ODA' AND ISNULL(A.CurrentLimit,0)=0 and ContiExcessDt IS
        */

        IF p_TIMEKEY > 26267 THEN
            UPDATE PRO.AccountCal_Stg A
            SET A.DPD_PrincOverdue    = (CASE WHEN A.PrincOverdueSinceDt IS NOT NULL THEN (v_ProcessDate - A.PrincOverdueSinceDt) + 1 ELSE 0 END),
                A.DPD_IntOverdueSince = (CASE WHEN A.IntOverdueSinceDt IS NOT NULL THEN (v_ProcessDate - A.IntOverdueSinceDt) + 1 ELSE 0 END),
                A.DPD_OtherOverdueSince = (CASE WHEN A.OtherOverdueSinceDt IS NOT NULL THEN (v_ProcessDate - A.OtherOverdueSinceDt) + 1 ELSE 0 END);
        ELSE
            UPDATE PRO.AccountCal_Stg A
            SET A.DPD_PrincOverdue    = (CASE WHEN A.PrincOverdueSinceDt IS NOT NULL THEN (v_ProcessDate - A.PrincOverdueSinceDt) ELSE 0 END),
                A.DPD_IntOverdueSince = (CASE WHEN A.IntOverdueSinceDt IS NOT NULL THEN (v_ProcessDate - A.IntOverdueSinceDt) ELSE 0 END),
                A.DPD_OtherOverdueSince = (CASE WHEN A.OtherOverdueSinceDt IS NOT NULL THEN (v_ProcessDate - A.OtherOverdueSinceDt) ELSE 0 END);
        END IF;

        /*--------------IF ANY DPD IS NEGATIVE THEN ZERO---------------------------------*/
        UPDATE PRO.AccountCal_Stg SET DPD_IntService      = 0 WHERE NVL(DPD_IntService,0)      < 0;
        UPDATE PRO.AccountCal_Stg SET DPD_NoCredit        = 0 WHERE NVL(DPD_NoCredit,0)        < 0;
        UPDATE PRO.AccountCal_Stg SET DPD_Overdrawn       = 0 WHERE NVL(DPD_Overdrawn,0)       < 0;
        UPDATE PRO.AccountCal_Stg SET DPD_Overdue         = 0 WHERE NVL(DPD_Overdue,0)         < 0;
        UPDATE PRO.AccountCal_Stg SET DPD_Renewal         = 0 WHERE NVL(DPD_Renewal,0)         < 0;
        UPDATE PRO.AccountCal_Stg SET DPD_StockStmt       = 0 WHERE NVL(DPD_StockStmt,0)       < 0;
        UPDATE PRO.AccountCal_Stg SET DPD_PrincOverdue    = 0 WHERE NVL(DPD_PrincOverdue,0)    < 0;
        UPDATE PRO.AccountCal_Stg SET DPD_IntOverdueSince = 0 WHERE NVL(DPD_IntOverdueSince,0) < 0;
        UPDATE PRO.AccountCal_Stg SET DPD_OtherOverdueSince = 0 WHERE NVL(DPD_OtherOverdueSince,0) < 0;

        /*------------DPD IS ZERO FOR ALL CC ACCOUNT DUE TO LASTCRDATE ------------------------------------*/
        --UPDATE A SET DPD_NoCredit=0
        --FROM PRO.AccountCal A INNER JOIN PRO.CustomerCal B ON A.RefCustomerID=B.RefCustomerID
        --WHERE isnull(B.FlgProcessing,'N')='N'

        /* RESTR WORK */

        MERGE INTO PRO.AdvAcRestructureCal T
        USING (
            SELECT AccountEntityID, MAX(DPD) DPD_MaxFin
            FROM (
                SELECT AccountEntityID, DPD_IntService DPD FROM PRO.AccountCal_Stg WHERE NVL(DPD_IntService,0) > 0
                UNION ALL SELECT AccountEntityID, DPD_NoCredit  DPD FROM PRO.AccountCal_Stg WHERE NVL(DPD_NoCredit,0)  > 0
                UNION ALL SELECT AccountEntityID, DPD_Overdrawn DPD FROM PRO.AccountCal_Stg WHERE NVL(DPD_Overdrawn,0) > 0
                UNION ALL SELECT AccountEntityID, DPD_Overdue   DPD FROM PRO.AccountCal_Stg WHERE NVL(DPD_Overdue,0)   > 0
            )
            GROUP BY AccountEntityID
        ) A ON (T.AccountEntityId = A.AccountEntityID)
        WHEN MATCHED THEN UPDATE SET T.DPD_MaxFin = A.DPD_MaxFin;

        MERGE INTO PRO.AdvAcRestructureCal T
        USING (
            SELECT AccountEntityID, MAX(DPD) DPD_MaxNonFin
            FROM (
                SELECT AccountEntityID, DPD_StockStmt DPD FROM PRO.AccountCal_Stg WHERE NVL(DPD_StockStmt,0) > 0
                UNION ALL SELECT AccountEntityID, DPD_Renewal DPD FROM PRO.AccountCal_Stg WHERE NVL(DPD_Renewal,0) > 0
            )
            GROUP BY AccountEntityID
        ) A ON (T.AccountEntityId = A.AccountEntityID)
        WHEN MATCHED THEN UPDATE SET T.DPD_MaxNonFin = A.DPD_MaxNonFin;

        UPDATE PRO.AdvAcRestructureCal SET DPD_MaxNonFin = 0 WHERE DPD_MaxNonFin IS NULL;
        UPDATE PRO.AdvAcRestructureCal SET DPD_MaxFin    = 0 WHERE DPD_MaxNonFin IS NULL;
        /* END OF RETR */

        --------------------------DPD UPDATE FOR Aqua Scheme---Prashant under guidence of Akshay Sir----03122025---------------
        MERGE INTO PRO.AccountCal_Stg T
        USING (
            SELECT A.AccountEntityID
            FROM PRO.AccountCal_Stg A
            INNER JOIN DIMPRODUCT C ON A.ProductAlt_Key = C.ProductAlt_Key
            WHERE C.EffectiveFromTimeKey <= p_TIMEKEY AND C.EffectiveToTimeKey >= p_TIMEKEY
              AND (NVL(C.Aqua_Scheme,'N') = 'Y' AND NVL(C.SchemeType,'') = 'ODA')
        ) S ON (T.AccountEntityID = S.AccountEntityID)
        WHEN MATCHED THEN UPDATE SET
            T.IntNotServicedDt = NULL,
            T.DPD_IntService   = 0,
            T.DebitSinceDt     = NULL,
            T.LastCrDate       = NULL,
            T.DPD_NoCredit     = 0,
            T.StockStDt        = NULL,
            T.DPD_StockStmt    = 0;
        --------------------END------DPD UPDATE FOR Aqua Scheme---Prashant under guidence of Akshay Sir----03122025---------------

        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED     = 'Y',
            ERRORDATE     = NULL,
            ERRORDESCRIPTION = NULL,
            "COUNT"       = NVL("COUNT",0) + 1
        WHERE RUNNINGPROCESSNAME = 'DPD_Calculation';

        -----------------Added for DashBoard 04-03-2021
        --Update BANDAUDITSTATUS set CompletedCount=CompletedCount+1 where BandName='ASSET CLASSIFICATION'

    EXCEPTION
        WHEN OTHERS THEN
            UPDATE PRO.ACLRUNNINGPROCESSSTATUS
            SET COMPLETED     = 'N',
                ERRORDATE     = SYSDATE,
                ERRORDESCRIPTION = SQLERRM,
                "COUNT"       = NVL("COUNT",0) + 1
            WHERE RUNNINGPROCESSNAME = 'DPD_Calculation';
    END;

END PRO.DPD_Calculation;
/
