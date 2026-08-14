CREATE OR REPLACE PROCEDURE PRO.NPA_Date_Calculation(
    p_TIMEKEY IN NUMBER
)
AS
/*=========================================
 AUTHER : TRILOKI KHANNA
 CREATE DATE : 27-11-2019
 MODIFY DATE : 27-11-2019
 DESCRIPTION : CALCULATED NPA DATE
 --EXEC [PRO].[NPA_DATE_CALCULATION]  @TIMEKEY=25841
=============================================*/
    v_INTTSERNORM  VARCHAR2(50);
    v_ProcessDate  DATE;
    v_error        VARCHAR2(4000);
BEGIN

    BEGIN
        SELECT REFVALUE INTO v_INTTSERNORM
        FROM PRO.REFPERIOD
        WHERE BUSINESSRULE='RECOVERYADJUSTMENT'
          AND EffectiveFromTimeKey<=p_TIMEKEY
          AND EffectiveToTimeKey>=p_TIMEKEY
          AND ROWNUM=1;
    EXCEPTION WHEN NO_DATA_FOUND THEN v_INTTSERNORM := NULL;
    END;

    BEGIN
        SELECT "Date" INTO v_ProcessDate
        FROM SysDayMatrix
        WHERE TimeKey=p_TIMEKEY
          AND ROWNUM=1;
    EXCEPTION WHEN NO_DATA_FOUND THEN v_ProcessDate := NULL;
    END;

    /* ------------------------------------------------------------------ */
    /* Null out sentinel NPA dates                                         */
    /* ------------------------------------------------------------------ */

    UPDATE PRO.AccountCal_Stg SET InitialNpaDt=NULL
    WHERE InitialNpaDt=DATE '1900-01-01'
       OR InitialNpaDt=TO_DATE('01/01/1900','MM/DD/YYYY');

    UPDATE PRO.AccountCal_Stg SET FinalNpaDt=NULL
    WHERE FinalNpaDt=DATE '1900-01-01'
       OR FinalNpaDt=TO_DATE('01/01/1900','MM/DD/YYYY');

    UPDATE PRO.AccountCal_Stg A
    SET A.InitialNpaDt=NULL, A.FinalNpaDt=NULL
    WHERE EXISTS (
        SELECT 1 FROM PRO.CustomerCal_Stg B
        WHERE A.REFCUSTOMERID=B.REFCUSTOMERID
          AND NVL(B.FlgProcessing,'N')='N'
    )
    AND NVL(A.FLGDEG,'N')='Y';

    /* ------------------------------------------------------------------ */
    /* CALCULATE NpaDt                                                     */
    /* #TEMPTABLEDPD and its mutation folded inline into the DPD_NoCredit  */
    /* CASE expression; #TEMPTABLENPA folded into the MERGE source.        */
    /* ------------------------------------------------------------------ */

    MERGE INTO PRO.AccountCal_Stg TGT
    USING (
        SELECT TGT2.ROWID AS RID,
               v_ProcessDate - NVL(NPA.REFPERIODNPA,0) + NVL(TGT2.REFPERIODMAX,0) AS NEW_FINALNPADT
        FROM (
            /* #TEMPTABLENPA – MAX-DPD selector */
            SELECT TT.CustomerAcID,
                   CASE
                       WHEN (NVL(TT.DPD_IntService,0)>=NVL(TT.DPD_NoCredit,0)
                             AND NVL(TT.DPD_IntService,0)>=NVL(TT.DPD_Overdrawn,0)
                             AND NVL(TT.DPD_IntService,0)>=NVL(TT.DPD_Overdue,0)
                             AND NVL(TT.DPD_IntService,0)>=NVL(TT.DPD_Renewal,0)
                             AND NVL(TT.DPD_IntService,0)>=NVL(TT.DPD_StockStmt,0))
                           THEN NVL(TT.DPD_IntService,0)
                       WHEN (NVL(TT.DPD_NoCredit,0)>=NVL(TT.DPD_IntService,0)
                             AND NVL(TT.DPD_NoCredit,0)>=NVL(TT.DPD_Overdrawn,0)
                             AND NVL(TT.DPD_NoCredit,0)>=NVL(TT.DPD_Overdue,0)
                             AND NVL(TT.DPD_NoCredit,0)>=NVL(TT.DPD_Renewal,0)
                             AND NVL(TT.DPD_NoCredit,0)>=NVL(TT.DPD_StockStmt,0))
                           THEN NVL(TT.DPD_NoCredit,0)
                       WHEN (NVL(TT.DPD_Overdrawn,0)>=NVL(TT.DPD_NoCredit,0)
                             AND NVL(TT.DPD_Overdrawn,0)>=NVL(TT.DPD_IntService,0)
                             AND NVL(TT.DPD_Overdrawn,0)>=NVL(TT.DPD_Overdue,0)
                             AND NVL(TT.DPD_Overdrawn,0)>=NVL(TT.DPD_Renewal,0)
                             AND NVL(TT.DPD_Overdrawn,0)>=NVL(TT.DPD_StockStmt,0))
                           THEN NVL(TT.DPD_Overdrawn,0)
                       WHEN (NVL(TT.DPD_Renewal,0)>=NVL(TT.DPD_NoCredit,0)
                             AND NVL(TT.DPD_Renewal,0)>=NVL(TT.DPD_IntService,0)
                             AND NVL(TT.DPD_Renewal,0)>=NVL(TT.DPD_Overdrawn,0)
                             AND NVL(TT.DPD_Renewal,0)>=NVL(TT.DPD_Overdue,0)
                             AND NVL(TT.DPD_Renewal,0)>=NVL(TT.DPD_StockStmt,0))
                           THEN NVL(TT.DPD_Renewal,0)
                       WHEN (NVL(TT.DPD_Overdue,0)>=NVL(TT.DPD_NoCredit,0)
                             AND NVL(TT.DPD_Overdue,0)>=NVL(TT.DPD_IntService,0)
                             AND NVL(TT.DPD_Overdue,0)>=NVL(TT.DPD_Overdrawn,0)
                             AND NVL(TT.DPD_Overdue,0)>=NVL(TT.DPD_Renewal,0)
                             AND NVL(TT.DPD_Overdue,0)>=NVL(TT.DPD_StockStmt,0))
                           THEN NVL(TT.DPD_Overdue,0)
                       ELSE NVL(TT.DPD_StockStmt,0)
                   END AS REFPERIODNPA
            FROM (
                /* #TEMPTABLEDPD – threshold filter with DPD_NoCredit mutation folded in */
                SELECT A.CustomerAcID,
                       CASE WHEN NVL(A.DPD_IntService,0)>=NVL(A.RefPeriodIntService,0)
                            THEN A.DPD_IntService
                            ELSE 0
                       END DPD_IntService,
                       CASE WHEN NVL(A.DPD_NoCredit,0)>=NVL(A.RefPeriodNoCredit,0)
                            THEN
                                CASE WHEN NVL(A.FLGDEG,'N')='Y'
                                          AND NVL(A.SourceAlt_Key,0)=1
                                          AND NVL(A.RefPeriodNoCredit,91)=91
                                          AND NVL(A.FacilityType,'') IN ('CC','OD')
                                          AND NVL(DP.SchemeType,'')='ODA'
                                          AND DP.ProductCode NOT IN ('DLFIN','VFVEN')
                                          AND NVL(DP.Aqua_Scheme,'N')='N'
                                     THEN LEAST(A.DPD_NoCredit, NVL(A.RefPeriodNoCredit,0))
                                     ELSE A.DPD_NoCredit
                                END
                            ELSE 0
                       END DPD_NoCredit,
                       CASE WHEN NVL(A.DPD_Overdrawn,0)>=NVL(A.RefPeriodOverDrawn,0)
                            THEN A.DPD_Overdrawn
                            ELSE 0
                       END DPD_Overdrawn,
                       CASE WHEN NVL(A.DPD_Overdue,0)>=NVL(A.RefPeriodOverdue,0)
                            THEN A.DPD_Overdue
                            ELSE 0
                       END DPD_Overdue,
                       CASE WHEN NVL(A.DPD_Renewal,0)>=NVL(A.RefPeriodReview,0)
                            THEN A.DPD_Renewal
                            ELSE 0
                       END DPD_Renewal,
                       CASE WHEN NVL(A.DPD_StockStmt,0)>=NVL(A.RefPeriodStkStatement,0)
                            THEN A.DPD_StockStmt
                            ELSE 0
                       END DPD_StockStmt
                FROM PRO.AccountCal_Stg A
                LEFT JOIN DimProduct DP
                       ON A.ProductAlt_Key=DP.ProductAlt_Key
                      AND DP.EffectiveFromTimeKey<=p_TIMEKEY
                      AND DP.EffectiveToTimeKey>=p_TIMEKEY
                WHERE (NVL(A.DPD_IntService,0)>=NVL(A.RefPeriodIntService,0)
                    OR NVL(A.DPD_NoCredit,0)>=NVL(A.RefPeriodNoCredit,0)
                    OR NVL(A.DPD_Overdrawn,0)>=NVL(A.RefPeriodOverDrawn,0)
                    OR NVL(A.DPD_Overdue,0)>=NVL(A.RefPeriodOverdue,0)
                    OR NVL(A.DPD_Renewal,0)>=NVL(A.RefPeriodReview,0)
                    OR NVL(A.DPD_StockStmt,0)>=NVL(A.RefPeriodStkStatement,0))
            ) TT
        ) NPA
        INNER JOIN PRO.AccountCal_Stg TGT2 ON NPA.CustomerAcID=TGT2.CustomerAcID
        WHERE NVL(TGT2.FLGDEG,'N')='Y'
    ) SRC ON (TGT.ROWID=SRC.RID)
    WHEN MATCHED THEN UPDATE SET TGT.FinalNpaDt=SRC.NEW_FINALNPADT;

    /* ------------------------------------------------------------------ */
    /* ALWYS_NPA – set FinalNpaDt to process date                         */
    /* ------------------------------------------------------------------ */

    UPDATE PRO.AccountCal_Stg A
    SET A.FINALNPADT=v_ProcessDate
    WHERE EXISTS (
        SELECT 1 FROM PRO.CustomerCal_Stg B
        WHERE A.REFCUSTOMERID=B.REFCUSTOMERID
    )
    AND A.ASSET_NORM='ALWYS_NPA'
    AND NVL(A.FLGDEG,'N')='Y';

    /* ------------------------------------------------------------------ */
    /* EXCEPTIONAL UPDATE FOR NPA DATE FOR EXISTING NPA ACCOUNT           */
    /* Update AdvAcRestructureCal.DEGDATE; includes SP_Expiry condition   */
    /* added by prashant 02082025                                          */
    /* ------------------------------------------------------------------ */

    MERGE INTO PRO.AdvAcRestructureCal B
    USING (
        SELECT B.ROWID AS RID,
               CASE WHEN B.PreRestructureNPA_Date IS NOT NULL
                    THEN B.PreRestructureNPA_Date
                    ELSE B.RestructureDt
               END AS NEW_DEGDATE
        FROM PRO.AccountCal_Stg A
        INNER JOIN PRO.AdvAcRestructureCal B ON A.AccountEntityID=B.AccountEntityId
        WHERE (A.FINALASSETCLASSALT_KEY>1 OR A.FlgDeg='Y')
          AND (CASE WHEN NVL(B.SP_ExpiryDate,DATE '1900-01-01')>=NVL(B.SP_ExpiryExtendedDate,DATE '1900-01-01')
                    THEN B.SP_ExpiryDate
                    ELSE B.SP_ExpiryExtendedDate
               END) > v_ProcessDate
    ) SRC ON (B.ROWID=SRC.RID)
    WHEN MATCHED THEN UPDATE SET B.DEGDATE=SRC.NEW_DEGDATE;

    /* ------------------------------------------------------------------ */
    /* #RESTR_NPA folded inline (first use) – update CustomerCal_Stg      */
    /* ------------------------------------------------------------------ */

    MERGE INTO PRO.CustomerCal_Stg C
    USING (
        SELECT A.UcifEntityID,
               MIN(CASE WHEN NVL(A.FinalNpaDt,DATE '2099-12-31')>NVL(B.DegDate,DATE '2099-12-31')
                        THEN B.DegDate
                        ELSE A.FinalNpaDt
               END) AS FinalNpaDt
        FROM PRO.AccountCal_Stg A
        INNER JOIN PRO.AdvAcRestructureCal B ON A.AccountEntityID=B.AccountEntityId
        INNER JOIN DimParameter D
               ON D.EffectiveFromTimeKey<=p_TIMEKEY
              AND D.EffectiveToTimeKey>=p_TIMEKEY
              AND D.ParameterAlt_Key=B.RestructureTypeAlt_Key
              AND D.DimParameterName='TypeofRestructuring'
        WHERE (A.FINALASSETCLASSALT_KEY>1 OR A.FlgDeg='Y')
        GROUP BY A.UcifEntityID
    ) SRC ON (C.UcifEntityID=SRC.UcifEntityID
          AND (C.SysAssetClassAlt_Key>1 OR C.FlgDeg='Y'))
    WHEN MATCHED THEN UPDATE SET C.SysNPA_Dt=NVL(SRC.FinalNpaDt,C.SysNPA_Dt);

    /* ------------------------------------------------------------------ */
    /* #RESTR_NPA folded inline (second use) – update AccountCal_Stg      */
    /* ------------------------------------------------------------------ */

    MERGE INTO PRO.AccountCal_Stg C
    USING (
        SELECT A.UcifEntityID,
               MIN(CASE WHEN NVL(A.FinalNpaDt,DATE '2099-12-31')>NVL(B.DegDate,DATE '2099-12-31')
                        THEN B.DegDate
                        ELSE A.FinalNpaDt
               END) AS FinalNpaDt
        FROM PRO.AccountCal_Stg A
        INNER JOIN PRO.AdvAcRestructureCal B ON A.AccountEntityID=B.AccountEntityId
        INNER JOIN DimParameter D
               ON D.EffectiveFromTimeKey<=p_TIMEKEY
              AND D.EffectiveToTimeKey>=p_TIMEKEY
              AND D.ParameterAlt_Key=B.RestructureTypeAlt_Key
              AND D.DimParameterName='TypeofRestructuring'
        WHERE (A.FINALASSETCLASSALT_KEY>1 OR A.FlgDeg='Y')
        GROUP BY A.UcifEntityID
    ) SRC ON (C.UcifEntityID=SRC.UcifEntityID
          AND (C.FINALASSETCLASSALT_KEY>1 OR C.FlgDeg='Y'))
    WHEN MATCHED THEN UPDATE SET C.FinalNpaDt=SRC.FinalNpaDt;

    /* ------------------------------------------------------------------ */
    /* Reset FlgDeg on non-qualifying restructure types                   */
    /* ------------------------------------------------------------------ */

    UPDATE PRO.AdvAcRestructureCal A
    SET A.FlgDeg='N'
    WHERE A.FlgDeg='Y'
      AND EXISTS (
          SELECT 1 FROM DimParameter D
          WHERE D.EffectiveFromTimeKey<=p_TIMEKEY
            AND D.EffectiveToTimeKey>=p_TIMEKEY
            AND D.ParameterAlt_Key=A.RestructureTypeAlt_Key
            AND D.DimParameterName='TypeofRestructuring'
            AND D.ParameterShortNameEnum NOT IN ('PRUDENTIAL','IRAC','OTHER')
      );

    /* ------------------------------------------------------------------ */
    /* PUI WORK                                                            */
    /* ------------------------------------------------------------------ */

    /* PUI (a) – Flg_Deg='Y' accounts: set FlgDeg, FinalNpaDt, DegReason, Asset_Norm */

    MERGE INTO PRO.AccountCal_Stg A
    USING (
        SELECT B.AccountEntityId,
               B.FLG_DEG,
               B.NPA_DATE,
               B.DEFAULT_REASON,
               B.Asset_Norm
        FROM PRO.PUI_CAL B
        WHERE B.Flg_Deg='Y'
    ) SRC ON (A.AccountEntityID=SRC.AccountEntityId)
    WHEN MATCHED THEN UPDATE SET
        A.FlgDeg    = SRC.FLG_DEG,
        A.FinalNpaDt= SRC.NPA_DATE,
        A.DegReason = NVL(A.DegReason,'')||','||SRC.DEFAULT_REASON,
        A.Asset_Norm= SRC.Asset_Norm;

    /* PUI (b) – Flg_Deg='N' accounts not already degraded and not ALWYS_STD */

    MERGE INTO PRO.AccountCal_Stg A
    USING (
        SELECT B.AccountEntityId,
               B.FLG_DEG,
               B.NPA_DATE,
               B.DEFAULT_REASON,
               B.Asset_Norm
        FROM PRO.PUI_CAL B
        WHERE B.Flg_Deg='N'
    ) SRC ON (A.AccountEntityID=SRC.AccountEntityId
          AND NVL(A.FlgDeg,'')<>'Y'
          AND NVL(SRC.Asset_Norm,'')<>'ALWYS_STD')
    WHEN MATCHED THEN UPDATE SET
        A.FlgDeg    = SRC.FLG_DEG,
        A.FinalNpaDt= SRC.NPA_DATE,
        A.NPA_REASON= NVL(A.NPA_REASON, SRC.DEFAULT_REASON),
        A.Asset_Norm= SRC.Asset_Norm;

    /* PUI (c) – CustomerCal_Stg: aggregate MIN NPA_DATE and LISTAGG reason from PUI_CAL */

    MERGE INTO PRO.CustomerCal_Stg D
    USING (
        SELECT A.CustomerEntityId,
               MIN(A.NPA_DATE)    AS NPA_DATE,
               'Y'                AS FLG_DEG,
               LISTAGG(A.DEFAULT_REASON, ', ')
                   WITHIN GROUP (ORDER BY A.DEFAULT_REASON) AS DEFAULT_REASON
        FROM PRO.PUI_CAL A
        INNER JOIN PRO.CustomerCal_Stg B ON A.CustomerEntityId=B.CustomerEntityId
        WHERE NVL(A.FLG_DEG,'N')='Y'
          AND NVL(B.FlgProcessing,'N')='N'
        GROUP BY A.CustomerEntityId
    ) SRC ON (D.CustomerEntityID=SRC.CustomerEntityId)
    WHEN MATCHED THEN UPDATE SET
        D.FlgDeg   = SRC.FLG_DEG,
        D.SysNPA_Dt= SRC.NPA_DATE,
        D.DegReason= NVL(D.DegReason,'')||','||SRC.DEFAULT_REASON;

    /* ------------------------------------------------------------------ */
    /* MIN NPA DATE CUSTOMER LEVEL                                         */
    /* ------------------------------------------------------------------ */

    MERGE INTO PRO.CustomerCal_Stg A
    USING (
        SELECT AC.REFCUSTOMERID,
               MIN(AC.FinalNpaDt) AS FinalNpaDt
        FROM PRO.AccountCal_Stg AC
        INNER JOIN PRO.CustomerCal_Stg BC ON AC.REFCUSTOMERID=BC.REFCUSTOMERID
        WHERE NVL(AC.FlgDeg,'N')='Y'
          AND NVL(BC.FlgProcessing,'N')='N'
        GROUP BY AC.REFCUSTOMERID
    ) C ON (A.REFCUSTOMERID=C.REFCUSTOMERID
        AND NVL(A.FlgProcessing,'N')='N')
    WHEN MATCHED THEN UPDATE SET A.SysNPA_Dt=C.FinalNpaDt, A.FlgDeg='Y';

    /* ------------------------------------------------------------------ */
    /* Final propagation: AccountCal_Stg.FINALNPADT = CustomerCal_Stg.SysNPA_Dt */
    /* ------------------------------------------------------------------ */

    MERGE INTO PRO.AccountCal_Stg A
    USING (
        SELECT A.ROWID AS RID, B.SysNPA_Dt
        FROM PRO.AccountCal_Stg A
        INNER JOIN PRO.CustomerCal_Stg B ON A.REFCUSTOMERID=B.REFCUSTOMERID
        WHERE NVL(A.ASSET_NORM,'NORMAL')<>'ALWYS_STD'
          AND NVL(A.FlgDeg,'N')='Y'
          AND NVL(B.FlgProcessing,'N')='N'
    ) SRC ON (A.ROWID=SRC.RID)
    WHEN MATCHED THEN UPDATE SET A.FINALNPADT=SRC.SysNPA_Dt;

    /* ------------------------------------------------------------------ */
    /* Mark process as complete                                            */
    /* ------------------------------------------------------------------ */

    UPDATE PRO.ACLRUNNINGPROCESSSTATUS
    SET COMPLETED='Y',
        ERRORDATE=NULL,
        ERRORDESCRIPTION=NULL,
        "COUNT"=NVL("COUNT",0)+1
    WHERE RUNNINGPROCESSNAME='NPA_Date_Calculation';

EXCEPTION
    WHEN OTHERS THEN
        v_error := SQLERRM;
        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED='N',
            ERRORDATE=SYSDATE,
            ERRORDESCRIPTION=v_error,
            "COUNT"=NVL("COUNT",0)+1
        WHERE RUNNINGPROCESSNAME='NPA_Date_Calculation';
END;
/
