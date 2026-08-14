CREATE OR REPLACE PROCEDURE PRO.MaxDPD_ReferencePeriod_Calculation(
    p_TIMEKEY IN NUMBER
)
AS
/*=========================================================
 AUTHER : TRILOKI KHANNA
 CREATE DATE : 27-11-2019
 MODIFY DATE : 27-11-2019
 DESCRIPTION : CALCULATED MAX DPD AND REGARD OF REFERENCE PERIOD ON  MAX DPD
 --EXEC [Pro].[MaxDPD_ReferencePeriod_Calculation] 25140
==============================================================*/
    v_error VARCHAR2(4000);
BEGIN

    /*--------------INTIAL MAX DPD 0 FOR RE PROCESSING DATA-------------------------*/

    UPDATE PRO.AccountCal_Stg A SET A.DPD_Max=0;

    /*----------------FIND MAX DPD---------------------------------------*/

    MERGE INTO PRO.AccountCal_Stg A
    USING (
        SELECT A.ROWID AS RID,
               (CASE
                   WHEN (NVL(A.DPD_IntService,0)>=NVL(A.DPD_NoCredit,0)   AND NVL(A.DPD_IntService,0)>=NVL(A.DPD_Overdrawn,0)
                         AND NVL(A.DPD_IntService,0)>=NVL(A.DPD_Overdue,0) AND NVL(A.DPD_IntService,0)>=NVL(A.DPD_Renewal,0)
                         AND NVL(A.DPD_IntService,0)>=NVL(A.DPD_StockStmt,0))
                       THEN NVL(A.DPD_IntService,0)
                   WHEN (NVL(A.DPD_NoCredit,0)>=NVL(A.DPD_IntService,0)   AND NVL(A.DPD_NoCredit,0)>=NVL(A.DPD_Overdrawn,0)
                         AND NVL(A.DPD_NoCredit,0)>=NVL(A.DPD_Overdue,0)  AND NVL(A.DPD_NoCredit,0)>=NVL(A.DPD_Renewal,0)
                         AND NVL(A.DPD_NoCredit,0)>=NVL(A.DPD_StockStmt,0))
                       THEN NVL(A.DPD_NoCredit,0)
                   WHEN (NVL(A.DPD_Overdrawn,0)>=NVL(A.DPD_NoCredit,0)    AND NVL(A.DPD_Overdrawn,0)>=NVL(A.DPD_IntService,0)
                         AND NVL(A.DPD_Overdrawn,0)>=NVL(A.DPD_Overdue,0) AND NVL(A.DPD_Overdrawn,0)>=NVL(A.DPD_Renewal,0)
                         AND NVL(A.DPD_Overdrawn,0)>=NVL(A.DPD_StockStmt,0))
                       THEN NVL(A.DPD_Overdrawn,0)
                   WHEN (NVL(A.DPD_Renewal,0)>=NVL(A.DPD_NoCredit,0)      AND NVL(A.DPD_Renewal,0)>=NVL(A.DPD_IntService,0)
                         AND NVL(A.DPD_Renewal,0)>=NVL(A.DPD_Overdrawn,0) AND NVL(A.DPD_Renewal,0)>=NVL(A.DPD_Overdue,0)
                         AND NVL(A.DPD_Renewal,0)>=NVL(A.DPD_StockStmt,0))
                       THEN NVL(A.DPD_Renewal,0)
                   WHEN (NVL(A.DPD_Overdue,0)>=NVL(A.DPD_NoCredit,0)      AND NVL(A.DPD_Overdue,0)>=NVL(A.DPD_IntService,0)
                         AND NVL(A.DPD_Overdue,0)>=NVL(A.DPD_Overdrawn,0) AND NVL(A.DPD_Overdue,0)>=NVL(A.DPD_Renewal,0)
                         AND NVL(A.DPD_Overdue,0)>=NVL(A.DPD_StockStmt,0))
                       THEN NVL(A.DPD_Overdue,0)
                   ELSE NVL(A.DPD_StockStmt,0)
               END) AS NEW_DPD_MAX
        FROM PRO.AccountCal_Stg A
        INNER JOIN PRO.CustomerCal_Stg C ON C.SourceSystemCustomerID=A.SourceSystemCustomerID
        WHERE NVL(C.FlgProcessing,'N')='N'
            AND (NVL(A.DPD_IntService,0)>0  OR NVL(A.DPD_Overdrawn,0)>0 OR NVL(A.DPD_Overdue,0)>0
                 OR NVL(A.DPD_Renewal,0)>0  OR NVL(A.DPD_StockStmt,0)>0 OR NVL(A.DPD_NoCredit,0)>0)
    ) SRC ON (A.ROWID=SRC.RID)
    WHEN MATCHED THEN UPDATE SET A.DPD_Max=SRC.NEW_DPD_MAX;

    /*----------------DPD_FinMaxType ---------------------------------------*/

    MERGE INTO PRO.AccountCal_Stg A
    USING (
        SELECT A.ROWID AS RID,
               (CASE
                   WHEN (NVL(A.DPD_IntService,0)>=NVL(A.DPD_NoCredit,0)   AND NVL(A.DPD_IntService,0)>=NVL(A.DPD_Overdrawn,0)
                         AND NVL(A.DPD_IntService,0)>=NVL(A.DPD_Overdue,0) AND NVL(A.DPD_IntService,0)>=NVL(A.DPD_Renewal,0)
                         AND NVL(A.DPD_IntService,0)>=NVL(A.DPD_StockStmt,0))
                       THEN 'RefPeriodIntService'
                   WHEN (NVL(A.DPD_NoCredit,0)>=NVL(A.DPD_IntService,0)   AND NVL(A.DPD_NoCredit,0)>=NVL(A.DPD_Overdrawn,0)
                         AND NVL(A.DPD_NoCredit,0)>=NVL(A.DPD_Overdue,0)  AND NVL(A.DPD_NoCredit,0)>=NVL(A.DPD_Renewal,0)
                         AND NVL(A.DPD_NoCredit,0)>=NVL(A.DPD_StockStmt,0))
                       THEN 'RefPeriodNoCredit'
                   WHEN (NVL(A.DPD_Overdrawn,0)>=NVL(A.DPD_NoCredit,0)    AND NVL(A.DPD_Overdrawn,0)>=NVL(A.DPD_IntService,0)
                         AND NVL(A.DPD_Overdrawn,0)>=NVL(A.DPD_Overdue,0) AND NVL(A.DPD_Overdrawn,0)>=NVL(A.DPD_Renewal,0)
                         AND NVL(A.DPD_Overdrawn,0)>=NVL(A.DPD_StockStmt,0))
                       THEN 'RefPeriodOverDrawn'
                   WHEN (NVL(A.DPD_Renewal,0)>=NVL(A.DPD_NoCredit,0)      AND NVL(A.DPD_Renewal,0)>=NVL(A.DPD_IntService,0)
                         AND NVL(A.DPD_Renewal,0)>=NVL(A.DPD_Overdrawn,0) AND NVL(A.DPD_Renewal,0)>=NVL(A.DPD_Overdue,0)
                         AND NVL(A.DPD_Renewal,0)>=NVL(A.DPD_StockStmt,0))
                       THEN 'RefPeriodReview'
                   WHEN (NVL(A.DPD_Overdue,0)>=NVL(A.DPD_NoCredit,0)      AND NVL(A.DPD_Overdue,0)>=NVL(A.DPD_IntService,0)
                         AND NVL(A.DPD_Overdue,0)>=NVL(A.DPD_Overdrawn,0) AND NVL(A.DPD_Overdue,0)>=NVL(A.DPD_Renewal,0)
                         AND NVL(A.DPD_Overdue,0)>=NVL(A.DPD_StockStmt,0))
                       THEN 'RefPeriodOverdue'
                   ELSE 'RefPeriodStkStatement'
               END) AS NEW_DPD_FINMAXTYPE
        FROM PRO.AccountCal_Stg A
        INNER JOIN PRO.CustomerCal_Stg C ON C.SourceSystemCustomerID=A.SourceSystemCustomerID
        WHERE NVL(C.FlgProcessing,'N')='N'
            AND (NVL(A.DPD_IntService,0)>0  OR NVL(A.DPD_Overdrawn,0)>0 OR NVL(A.DPD_Overdue,0)>0
                 OR NVL(A.DPD_Renewal,0)>0  OR NVL(A.DPD_StockStmt,0)>0 OR NVL(A.DPD_NoCredit,0)>0)
    ) SRC ON (A.ROWID=SRC.RID)
    WHEN MATCHED THEN UPDATE SET A.DPD_FinMaxType=SRC.NEW_DPD_FINMAXTYPE;

    /*-------Update REFPeriodMax---------------------------*/
    /*------- INTIAL REFPERIODMAX 0 FOR RE-PROCESSING----- */

    MERGE INTO PRO.AccountCal_Stg TGT
    USING (
        SELECT TGT2.ROWID AS RID
        FROM (
            SELECT A.CustomerAcID,
                   CASE WHEN NVL(A.DPD_IntService,0)>=NVL(A.RefPeriodIntService,0) THEN A.DPD_IntService ELSE 0 END DPD_IntService,
                   CASE WHEN NVL(A.DPD_NoCredit,0)>=NVL(A.RefPeriodNoCredit,0)     THEN A.DPD_NoCredit    ELSE 0 END DPD_NoCredit,
                   CASE WHEN NVL(A.DPD_Overdrawn,0)>=NVL(A.RefPeriodOverDrawn,0)   THEN A.DPD_Overdrawn   ELSE 0 END DPD_Overdrawn,
                   CASE WHEN NVL(A.DPD_Overdue,0)>=NVL(A.RefPeriodOverdue,0)       THEN A.DPD_Overdue     ELSE 0 END DPD_Overdue,
                   CASE WHEN NVL(A.DPD_Renewal,0)>=NVL(A.RefPeriodReview,0)        THEN A.DPD_Renewal     ELSE 0 END DPD_Renewal,
                   CASE WHEN NVL(A.DPD_StockStmt,0)>=NVL(A.RefPeriodStkStatement,0) THEN A.DPD_StockStmt  ELSE 0 END DPD_StockStmt
            FROM PRO.AccountCal_Stg A
            WHERE (NVL(A.DPD_IntService,0)>=NVL(A.RefPeriodIntService,0)
                   OR NVL(A.DPD_NoCredit,0)>=NVL(A.RefPeriodNoCredit,0)
                   OR NVL(A.DPD_Overdrawn,0)>=NVL(A.RefPeriodOverDrawn,0)
                   OR NVL(A.DPD_Overdue,0)>=NVL(A.RefPeriodOverdue,0)
                   OR NVL(A.DPD_Renewal,0)>=NVL(A.RefPeriodReview,0)
                   OR NVL(A.DPD_StockStmt,0)>=NVL(A.RefPeriodStkStatement,0))
        ) TT
        INNER JOIN PRO.AccountCal_Stg TGT2 ON TT.CustomerAcID=TGT2.CustomerAcID
        INNER JOIN PRO.CustomerCal_Stg C ON C.SourceSystemCustomerID=TGT2.SourceSystemCustomerID
        WHERE NVL(C.FLGPROCESSING,'N')='N'
    ) SRC ON (TGT.ROWID=SRC.RID)
    WHEN MATCHED THEN UPDATE SET TGT.REFPERIODMAX=0;

    /*----CALCULATE REFPERIODMAX  REGARDING MAX DPD--------------*/

    MERGE INTO PRO.AccountCal_Stg TGT
    USING (
        SELECT TGT2.ROWID AS RID,
               CASE
                   WHEN (NVL(TT.DPD_IntService,0)>=NVL(TT.DPD_NoCredit,0)   AND NVL(TT.DPD_IntService,0)>=NVL(TT.DPD_Overdrawn,0)
                         AND NVL(TT.DPD_IntService,0)>=NVL(TT.DPD_Overdue,0) AND NVL(TT.DPD_IntService,0)>=NVL(TT.DPD_Renewal,0)
                         AND NVL(TT.DPD_IntService,0)>=NVL(TT.DPD_StockStmt,0))
                       THEN NVL(TGT2.RefPeriodIntService,0)
                   WHEN (NVL(TT.DPD_NoCredit,0)>=NVL(TT.DPD_IntService,0)   AND NVL(TT.DPD_NoCredit,0)>=NVL(TT.DPD_Overdrawn,0)
                         AND NVL(TT.DPD_NoCredit,0)>=NVL(TT.DPD_Overdue,0)  AND NVL(TT.DPD_NoCredit,0)>=NVL(TT.DPD_Renewal,0)
                         AND NVL(TT.DPD_NoCredit,0)>=NVL(TT.DPD_StockStmt,0))
                       THEN NVL(TGT2.RefPeriodNoCredit,0)
                   WHEN (NVL(TT.DPD_Overdrawn,0)>=NVL(TT.DPD_NoCredit,0)    AND NVL(TT.DPD_Overdrawn,0)>=NVL(TT.DPD_IntService,0)
                         AND NVL(TT.DPD_Overdrawn,0)>=NVL(TT.DPD_Overdue,0) AND NVL(TT.DPD_Overdrawn,0)>=NVL(TT.DPD_Renewal,0)
                         AND NVL(TT.DPD_Overdrawn,0)>=NVL(TT.DPD_StockStmt,0))
                       THEN NVL(TGT2.RefPeriodOverDrawn,0)
                   WHEN (NVL(TT.DPD_Renewal,0)>=NVL(TT.DPD_NoCredit,0)      AND NVL(TT.DPD_Renewal,0)>=NVL(TT.DPD_IntService,0)
                         AND NVL(TT.DPD_Renewal,0)>=NVL(TT.DPD_Overdrawn,0) AND NVL(TT.DPD_Renewal,0)>=NVL(TT.DPD_Overdue,0)
                         AND NVL(TT.DPD_Renewal,0)>=NVL(TT.DPD_StockStmt,0))
                       THEN NVL(TGT2.RefPeriodReview,0)
                   WHEN (NVL(TT.DPD_Overdue,0)>=NVL(TT.DPD_NoCredit,0)      AND NVL(TT.DPD_Overdue,0)>=NVL(TT.DPD_IntService,0)
                         AND NVL(TT.DPD_Overdue,0)>=NVL(TT.DPD_Overdrawn,0) AND NVL(TT.DPD_Overdue,0)>=NVL(TT.DPD_Renewal,0)
                         AND NVL(TT.DPD_Overdue,0)>=NVL(TT.DPD_StockStmt,0))
                       THEN NVL(TGT2.RefPeriodOverdue,0)
                   ELSE NVL(TGT2.RefPeriodStkStatement,0)
               END AS REFPERIOD
        FROM (
            SELECT A.CustomerAcID,
                   CASE WHEN NVL(A.DPD_IntService,0)>=NVL(A.RefPeriodIntService,0) THEN A.DPD_IntService ELSE 0 END DPD_IntService,
                   CASE WHEN NVL(A.DPD_NoCredit,0)>=NVL(A.RefPeriodNoCredit,0)     THEN A.DPD_NoCredit    ELSE 0 END DPD_NoCredit,
                   CASE WHEN NVL(A.DPD_Overdrawn,0)>=NVL(A.RefPeriodOverDrawn,0)   THEN A.DPD_Overdrawn   ELSE 0 END DPD_Overdrawn,
                   CASE WHEN NVL(A.DPD_Overdue,0)>=NVL(A.RefPeriodOverdue,0)       THEN A.DPD_Overdue     ELSE 0 END DPD_Overdue,
                   CASE WHEN NVL(A.DPD_Renewal,0)>=NVL(A.RefPeriodReview,0)        THEN A.DPD_Renewal     ELSE 0 END DPD_Renewal,
                   CASE WHEN NVL(A.DPD_StockStmt,0)>=NVL(A.RefPeriodStkStatement,0) THEN A.DPD_StockStmt  ELSE 0 END DPD_StockStmt
            FROM PRO.AccountCal_Stg A
            WHERE (NVL(A.DPD_IntService,0)>=NVL(A.RefPeriodIntService,0)
                   OR NVL(A.DPD_NoCredit,0)>=NVL(A.RefPeriodNoCredit,0)
                   OR NVL(A.DPD_Overdrawn,0)>=NVL(A.RefPeriodOverDrawn,0)
                   OR NVL(A.DPD_Overdue,0)>=NVL(A.RefPeriodOverdue,0)
                   OR NVL(A.DPD_Renewal,0)>=NVL(A.RefPeriodReview,0)
                   OR NVL(A.DPD_StockStmt,0)>=NVL(A.RefPeriodStkStatement,0))
        ) TT
        INNER JOIN PRO.AccountCal_Stg TGT2 ON TT.CustomerAcID=TGT2.CustomerAcID
        INNER JOIN PRO.CustomerCal_Stg C ON C.SourceSystemCustomerID=TGT2.SourceSystemCustomerID
        WHERE NVL(C.FLGPROCESSING,'N')='N'
    ) SRC ON (TGT.ROWID=SRC.RID)
    WHEN MATCHED THEN UPDATE SET TGT.REFPERIODMAX=SRC.REFPERIOD;

    /*---FOR HANDING NULL REFERENCE PERIOD ----------------------*/

    UPDATE PRO.AccountCal_Stg A
    SET A.REFPeriodMax=NVL(A.RefPeriodOverdue,0)
    WHERE NVL(A.FlgDeg,'N')='Y' AND NVL(A.InitialAssetClassAlt_Key,1)=1
        AND A.Balance>0 AND NVL(A.REFPeriodMax,0)=0
        AND NVL(A.DPD_Max,0)<NVL(A.RefPeriodOverdue,0)
        AND A.FacilityType IN('TL','DL','BP','BD','PC');

    UPDATE PRO.AccountCal_Stg A
    SET A.REFPeriodMax=NVL(A.RefPeriodIntService,0)
    WHERE NVL(A.FlgDeg,'N')='Y' AND NVL(A.InitialAssetClassAlt_Key,1)=1
        AND A.Balance>0
        AND NVL(A.DPD_Max,0)<NVL(A.RefPeriodIntService,0)
        AND A.FacilityType IN('CC','OD') AND NVL(A.REFPeriodMax,0)=0;

    ----Added By Triloki 10/06/2021  But if ALL DPD ZERO than REFPeriodMax is null---

    UPDATE PRO.AccountCal_Stg SET REFPeriodMax=RefPeriodNoCredit    WHERE REFPeriodMax IS NULL AND DPD_FinMaxType='RefPeriodNoCredit';
    UPDATE PRO.AccountCal_Stg SET REFPeriodMax=RefPeriodOverdue     WHERE REFPeriodMax IS NULL AND DPD_FinMaxType='RefPeriodOverdue';
    UPDATE PRO.AccountCal_Stg SET REFPeriodMax=RefPeriodOverDrawn   WHERE REFPeriodMax IS NULL AND DPD_FinMaxType='RefPeriodOverDrawn';
    UPDATE PRO.AccountCal_Stg SET REFPeriodMax=RefPeriodStkStatement WHERE REFPeriodMax IS NULL AND DPD_FinMaxType='RefPeriodStkStatement';
    UPDATE PRO.AccountCal_Stg SET REFPeriodMax=RefPeriodReview      WHERE REFPeriodMax IS NULL AND DPD_FinMaxType='RefPeriodReview';

    UPDATE PRO.ACLRUNNINGPROCESSSTATUS
    SET COMPLETED='Y',ERRORDATE=NULL,ERRORDESCRIPTION=NULL,"COUNT"=NVL("COUNT",0)+1
    WHERE RUNNINGPROCESSNAME='MaxDPD_ReferencePeriod_Calculation';

    -----------------Added for DashBoard 04-03-2021
    --Update BANDAUDITSTATUS set CompletedCount=CompletedCount+1 where BandName='ASSET CLASSIFICATION'

EXCEPTION
    WHEN OTHERS THEN
        v_error := SQLERRM;
        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED='N',ERRORDATE=SYSDATE,ERRORDESCRIPTION=v_error,"COUNT"=NVL("COUNT",0)+1
        WHERE RUNNINGPROCESSNAME='MaxDPD_ReferencePeriod_Calculation';
END;
/
