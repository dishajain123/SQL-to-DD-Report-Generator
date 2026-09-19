USE [DEMO_MISDB]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

/*=========================================
 DESCRIPTION : Appropriates available government guarantee cover against
               each guaranteed account's provisioning requirement in
               priority order by asset class, stages and upserts the
               appropriation into the cover ledger, walks shortfall
               accounts to log an escalation, and records the remaining
               fund balance for the next run.
 EXEC PRO.Guarantee_Cover_Appropriation @TimeKey = 25140
=============================================*/
CREATE PROCEDURE PRO.Guarantee_Cover_Appropriation
    @TimeKey INT
AS
BEGIN
    SET NOCOUNT ON
    BEGIN TRY

        DECLARE @ProcessDate DATE = (SELECT [Date] FROM SysDayMatrix WHERE TimeKey = @TimeKey)
        DECLARE @AvailableCoverBalance DECIMAL(18,2) = (SELECT AvailableBalance FROM PRO.GuaranteeFund WHERE FundId = 'GOVT_CGF')
        DECLARE @FundRenewalDate DATE = (SELECT RenewalDate FROM PRO.GuaranteeFund WHERE FundId = 'GOVT_CGF')

        -- Rule 1: NULL check - accounts flagged guarantee-covered but with
        -- no provisioning requirement recorded at all cannot be assessed
        UPDATE A
        SET A.CoverRequestedAmount = NULL,
            A.CoverShortfallFlag = 'N'
        FROM PRO.LoanAccountCal A
        WHERE A.GuaranteeCoveredFlag = 'Y'
          AND A.ProvisionAmount IS NULL

        -- Rule 2: only guaranteed accounts with a positive provisioning
        -- requirement are candidates for appropriation
        UPDATE A
        SET A.CoverRequestedAmount = A.ProvisionAmount
        FROM PRO.LoanAccountCal A
        WHERE A.GuaranteeCoveredFlag = 'Y'
          AND A.ProvisionAmount IS NOT NULL
          AND A.ProvisionAmount > 0

        -- Rule 2: sequential IF/ELSE - the fund must not be within its
        -- renewal blackout window, and must have a positive balance
        IF @FundRenewalDate IS NOT NULL AND @ProcessDate >= DATEADD(DAY, -7, @FundRenewalDate)
        BEGIN
            UPDATE A
            SET A.CoverAppropriatedAmount = 0,
                A.CoverShortfallFlag = 'Y'
            FROM PRO.LoanAccountCal A
            WHERE A.GuaranteeCoveredFlag = 'Y'
              AND A.CoverRequestedAmount > 0
        END
        ELSE IF @AvailableCoverBalance IS NOT NULL AND @AvailableCoverBalance > 0
        BEGIN
            -- Rule 3: nested condition on both asset class and whether the
            -- fund can cover it in full - accounts in the worse asset
            -- classes are appropriated first when balance is insufficient
            UPDATE A
            SET A.CoverAppropriatedAmount =
                CASE
                    WHEN A.AssetClass IN ('DOUBTFUL', 'LOSS') THEN
                        CASE
                            WHEN A.CoverRequestedAmount <= @AvailableCoverBalance THEN A.CoverRequestedAmount
                            ELSE @AvailableCoverBalance
                        END
                    WHEN A.AssetClass = 'SUBSTANDARD' THEN
                        CASE
                            WHEN A.CoverRequestedAmount <= @AvailableCoverBalance * 0.5 THEN A.CoverRequestedAmount
                            ELSE @AvailableCoverBalance * 0.5
                        END
                    ELSE 0
                END
            FROM PRO.LoanAccountCal A
            WHERE A.GuaranteeCoveredFlag = 'Y'
              AND A.CoverRequestedAmount > 0
        END
        ELSE
        BEGIN
            -- Rule 4: no fund balance available this cycle - nothing is
            -- appropriated, and the shortfall is flagged
            UPDATE A
            SET A.CoverAppropriatedAmount = 0,
                A.CoverShortfallFlag = 'Y'
            FROM PRO.LoanAccountCal A
            WHERE A.GuaranteeCoveredFlag = 'Y'
              AND A.CoverRequestedAmount > 0
        END

        -- Rule 5: net provisioning required after appropriation
        UPDATE A
        SET A.NetProvisionAfterCover = A.ProvisionAmount - ISNULL(A.CoverAppropriatedAmount, 0)
        FROM PRO.LoanAccountCal A
        WHERE A.GuaranteeCoveredFlag = 'Y'

        IF OBJECT_ID('tempdb..#CoverLedgerStaging') IS NOT NULL
            DROP TABLE #CoverLedgerStaging

        CREATE TABLE #CoverLedgerStaging
        (
            AccountId               VARCHAR(20),
            CoverAppropriatedAmount DECIMAL(18,2),
            NetProvisionAfterCover  DECIMAL(18,2)
        )

        -- Rule 6: conditional INSERT - only accounts actually processed
        -- for cover this cycle are staged
        INSERT INTO #CoverLedgerStaging (AccountId, CoverAppropriatedAmount, NetProvisionAfterCover)
        SELECT AccountId, CoverAppropriatedAmount, NetProvisionAfterCover
        FROM PRO.LoanAccountCal
        WHERE GuaranteeCoveredFlag = 'Y'
          AND CoverRequestedAmount > 0

        -- Rule 7: upsert the staged appropriation into the cover ledger -
        -- refresh accounts already tracked, add new ones
        MERGE PRO.GuaranteeCoverLedger AS Target
        USING #CoverLedgerStaging AS Source
        ON Target.AccountId = Source.AccountId
        WHEN MATCHED THEN
            UPDATE SET
                Target.CoverAppropriatedAmount = Source.CoverAppropriatedAmount,
                Target.NetProvisionAfterCover = Source.NetProvisionAfterCover,
                Target.LastAppropriationDate = @ProcessDate
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (AccountId, CoverAppropriatedAmount, NetProvisionAfterCover, FirstAppropriationDate, LastAppropriationDate)
            VALUES (Source.AccountId, Source.CoverAppropriatedAmount, Source.NetProvisionAfterCover, @ProcessDate, @ProcessDate);

        -- Rule 8: log an escalation for every shortfall account
        INSERT INTO PRO.CollectionsQueue (AccountId, EscalationDate, Reason)
        SELECT AccountId, @ProcessDate, 'GUARANTEE_COVER_SHORTFALL'
        FROM PRO.LoanAccountCal
        WHERE GuaranteeCoveredFlag = 'Y' AND CoverShortfallFlag = 'Y'

        -- Rule 9: record the remaining fund balance for the next run,
        -- summing appropriations made this cycle
        UPDATE PRO.GuaranteeFund
        SET AvailableBalance = @AvailableCoverBalance - (
                SELECT ISNULL(SUM(CoverAppropriatedAmount), 0)
                FROM PRO.LoanAccountCal
                WHERE GuaranteeCoveredFlag = 'Y'
            ),
            LastAppropriationDate = @ProcessDate
        WHERE FundId = 'GOVT_CGF'

        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'Y', ERRORDATE = NULL, ERRORDESCRIPTION = NULL, COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'Guarantee_Cover_Appropriation'

    END TRY
    BEGIN CATCH
        -- Exception handling: record the failure for operations to investigate
        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'N', ERRORDATE = GETDATE(), ERRORDESCRIPTION = ERROR_MESSAGE(), COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'Guarantee_Cover_Appropriation'
    END CATCH
    SET NOCOUNT OFF
END
GO
