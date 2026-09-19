USE [DEMO_MISDB]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

/*=========================================
 DESCRIPTION : Checks whether NPA accounts have cleared overdue for a
               full review period, marks them eligible (or not) for
               upgrade, performs the upgrade, calculates the provision
               released by the upgrade, and stamps the upgrade date.
 EXEC PRO.Asset_Class_Upgrade_Marking @TimeKey = 25140
=============================================*/
CREATE PROCEDURE PRO.Asset_Class_Upgrade_Marking
    @TimeKey INT
AS
BEGIN
    SET NOCOUNT ON
    BEGIN TRY

        DECLARE @ReviewPeriodDays INT = 90
        DECLARE @ProcessDate DATE = (SELECT [Date] FROM SysDayMatrix WHERE TimeKey = @TimeKey)

        -- Rule 1: mark NPA accounts eligible for upgrade if cleared for the full review period
        IF EXISTS (SELECT 1 FROM PRO.AccountCal WHERE AssetClass <> 'STANDARD')
        BEGIN
            UPDATE A
            SET A.UpgradeEligible = 'Y'
            FROM PRO.AccountCal A
            WHERE A.AssetClass <> 'STANDARD'
              AND A.OverdueDays = 0
              AND A.DaysSinceLastOverdue >= @ReviewPeriodDays
        END
        -- Rule 2: otherwise, mark all accounts as not eligible this run
        ELSE
        BEGIN
            UPDATE A
            SET A.UpgradeEligible = 'N'
            FROM PRO.AccountCal A
        END

        -- Rule 3 (formula, same balance * percent / 100 shape as the real
        -- AddlProvision calculation in PRO.UpdateNetBalance_AccountWise):
        -- calculate the provision released by the upgrade, before reclassifying
        UPDATE A
        SET A.ProvisionReleaseAmount = A.ProvisionAmount - ((A.OutstandingBalance * A.StandardProvisionPct) / 100)
        FROM PRO.AccountCal A
        WHERE A.UpgradeEligible = 'Y'

        -- Rule 4: reclassify eligible accounts back to Standard with base provisioning
        UPDATE A
        SET A.AssetClass = 'STANDARD',
            A.ProvisionPct = A.StandardProvisionPct,
            A.ProvisionAmount = (A.OutstandingBalance * A.StandardProvisionPct) / 100
        FROM PRO.AccountCal A
        WHERE A.UpgradeEligible = 'Y'

        -- Rule 5: stamp the date the upgrade took effect
        UPDATE A
        SET A.UpgradeDate = @ProcessDate
        FROM PRO.AccountCal A
        WHERE A.UpgradeEligible = 'Y'

        UPDATE PRO.RunStatus
        SET COMPLETED = 'Y', ErrorDate = NULL, ErrorDescription = NULL, RunCount = ISNULL(RunCount, 0) + 1
        WHERE ProcessName = 'Asset_Class_Upgrade_Marking'

    END TRY
    BEGIN CATCH
        -- Exception handling: record the failure for operations to investigate
        UPDATE PRO.RunStatus
        SET COMPLETED = 'N', ErrorDate = GETDATE(), ErrorDescription = ERROR_MESSAGE(), RunCount = ISNULL(RunCount, 0) + 1
        WHERE ProcessName = 'Asset_Class_Upgrade_Marking'
    END CATCH
END
GO
