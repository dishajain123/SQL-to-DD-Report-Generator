USE [DEMO_MISDB]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

/*=========================================
 DESCRIPTION : Evaluates each loan account for restructuring eligibility
               using nested conditions on asset class, overdue history,
               and prior restructuring count, stages and upserts the
               decision into the restructuring register, walks approved
               accounts to apply the revised terms, and audits every
               decision made this run.
 EXEC PRO.Loan_Restructuring_Eligibility @TimeKey = 25140
=============================================*/
CREATE PROCEDURE PRO.Loan_Restructuring_Eligibility
    @TimeKey INT
AS
BEGIN
    SET NOCOUNT ON
    BEGIN TRY

        DECLARE @ProcessDate DATE = (SELECT [Date] FROM SysDayMatrix WHERE TimeKey = @TimeKey)
        DECLARE @SchemeCutoffDate DATE = DATEADD(YEAR, -2, @ProcessDate)

        -- Rule 1: sequential IF/ELSE - the restructuring scheme is only
        -- open at all if today falls before the scheme cutoff
        IF @ProcessDate < @SchemeCutoffDate
        BEGIN
            -- Rule 2: nested IF/ELSE - only Standard or SMA accounts are
            -- considered; everything else is ineligible outright
            UPDATE A
            SET A.RestructureEligible =
                CASE
                    WHEN A.AssetClass IN ('STANDARD', 'SMA') THEN
                        CASE
                            -- Rule 3: within eligible asset classes, restructuring
                            -- count and overdue history further narrow eligibility
                            WHEN A.PriorRestructureCount IS NULL OR A.PriorRestructureCount = 0 THEN
                                CASE
                                    WHEN A.OverdueDays <= 60 THEN 'Y'
                                    ELSE 'N'
                                END
                            WHEN A.PriorRestructureCount = 1 THEN
                                CASE
                                    WHEN A.OverdueDays <= 30 AND A.OutstandingBalance IS NOT NULL THEN 'Y'
                                    ELSE 'N'
                                END
                            ELSE 'N'
                        END
                    ELSE 'N'
                END
            FROM PRO.LoanAccountCal A
        END
        ELSE
        BEGIN
            UPDATE A
            SET A.RestructureEligible = 'SCHEME_CLOSED'
            FROM PRO.LoanAccountCal A
        END

        -- Rule 4: accounts with no recorded outstanding balance cannot be
        -- assessed for restructuring at all and must be marked separately
        UPDATE A
        SET A.RestructureEligible = 'NOT_ASSESSED'
        FROM PRO.LoanAccountCal A
        WHERE A.OutstandingBalance IS NULL

        IF OBJECT_ID('tempdb..#RestructureDecisions') IS NOT NULL
            DROP TABLE #RestructureDecisions

        CREATE TABLE #RestructureDecisions
        (
            AccountId       VARCHAR(20),
            DecisionDate    DATE,
            EligibleFlag    VARCHAR(15),
            AssetClassAtEval VARCHAR(20),
            RevisedTenureMonths INT
        )

        -- Rule 5: conditional INSERT - capture the decision only for
        -- accounts that were actually evaluated
        INSERT INTO #RestructureDecisions (AccountId, DecisionDate, EligibleFlag, AssetClassAtEval, RevisedTenureMonths)
        SELECT A.AccountId, @ProcessDate, A.RestructureEligible, A.AssetClass, NULL
        FROM PRO.LoanAccountCal A
        WHERE A.RestructureEligible IS NOT NULL
          AND A.RestructureEligible <> 'NOT_ASSESSED'

        -- Rule 6: derived-assignment UPDATE reading back the staging table
        -- above - the revised tenure scales with the outstanding balance
        UPDATE D
        SET D.RevisedTenureMonths = (
                CASE
                    WHEN A.OutstandingBalance > 1000000 THEN 60
                    WHEN A.OutstandingBalance > 500000 THEN 48
                    ELSE 36
                END
            )
        FROM #RestructureDecisions D
        INNER JOIN PRO.LoanAccountCal A ON A.AccountId = D.AccountId
        WHERE D.EligibleFlag = 'Y'

        -- Rule 7: upsert every evaluated decision into the restructuring
        -- register - refresh accounts already tracked, add new ones
        MERGE PRO.RestructureRegister AS Target
        USING #RestructureDecisions AS Source
        ON Target.AccountId = Source.AccountId
        WHEN MATCHED THEN
            UPDATE SET
                Target.EligibleFlag = Source.EligibleFlag,
                Target.RevisedTenureMonths = Source.RevisedTenureMonths,
                Target.LastEvaluatedDate = @ProcessDate
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (AccountId, EligibleFlag, RevisedTenureMonths, FirstEvaluatedDate, LastEvaluatedDate)
            VALUES (Source.AccountId, Source.EligibleFlag, Source.RevisedTenureMonths, @ProcessDate, @ProcessDate);

        -- Rule 8: apply the revised terms to every approved account - a
        -- set-based status transition joined back to the staging table's
        -- own per-account tenure
        UPDATE A
        SET A.AccountStatus = 'RESTRUCTURED',
            A.LastPaymentDueDate = DATEADD(MONTH, D.RevisedTenureMonths, @ProcessDate),
            A.PriorRestructureCount = ISNULL(A.PriorRestructureCount, 0) + 1
        FROM PRO.LoanAccountCal A
        INNER JOIN #RestructureDecisions D ON D.AccountId = A.AccountId
        WHERE D.EligibleFlag = 'Y'

        -- Rule 9: persist only the accounts approved this run into the
        -- audit log, read back from the register populated above
        INSERT INTO PRO.RestructureAuditLog (AccountId, DecisionDate, EligibleFlag)
        SELECT R.AccountId, R.LastEvaluatedDate, R.EligibleFlag
        FROM PRO.RestructureRegister R
        WHERE R.LastEvaluatedDate = @ProcessDate
          AND R.EligibleFlag = 'Y'

        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'Y', ERRORDATE = NULL, ERRORDESCRIPTION = NULL, COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'Loan_Restructuring_Eligibility'

    END TRY
    BEGIN CATCH
        -- Exception handling: record the failure for operations to investigate
        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'N', ERRORDATE = GETDATE(), ERRORDESCRIPTION = ERROR_MESSAGE(), COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'Loan_Restructuring_Eligibility'
    END CATCH
    SET NOCOUNT OFF
END
GO
