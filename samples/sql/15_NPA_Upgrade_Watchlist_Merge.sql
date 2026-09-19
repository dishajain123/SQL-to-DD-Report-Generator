USE [DEMO_MISDB]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

/*=========================================
 DESCRIPTION : Stages NPA accounts that have cleared overdue for the
               mandatory watch period, upserts them into the upgrade
               watchlist, walks eligible accounts to promote them back
               to Standard asset class one at a time, and audits every
               promotion applied.
 EXEC PRO.NPA_Upgrade_Watchlist_Merge @TimeKey = 25140
=============================================*/
CREATE PROCEDURE PRO.NPA_Upgrade_Watchlist_Merge
    @TimeKey INT
AS
BEGIN
    SET NOCOUNT ON
    BEGIN TRY

        DECLARE @ProcessDate DATE = (SELECT [Date] FROM SysDayMatrix WHERE TimeKey = @TimeKey)
        DECLARE @StandardWatchPeriodDays INT = 365
        DECLARE @DoubtfulWatchPeriodDays INT = 545

        IF OBJECT_ID('tempdb..#WatchlistStaging') IS NOT NULL
            DROP TABLE #WatchlistStaging

        CREATE TABLE #WatchlistStaging
        (
            AccountId            VARCHAR(20),
            AssetClass           VARCHAR(20),
            DaysSinceLastOverdue INT,
            EligibleForUpgrade   VARCHAR(1)
        )

        -- Rule 1: date comparison/derivation - recompute how many days
        -- have elapsed since the account last cleared overdue, so the
        -- watch period check below is not relying on a stale column
        UPDATE A
        SET A.DaysSinceLastOverdue = DATEDIFF(DAY, A.LastOverdueClearedDate, @ProcessDate)
        FROM PRO.LoanAccountCal A
        WHERE A.AssetClass <> 'STANDARD'
          AND A.OverdueDays = 0
          AND A.LastOverdueClearedDate IS NOT NULL
          AND A.LastOverdueClearedDate <= @ProcessDate

        -- Rule 2: conditional INSERT - stage NPA accounts currently clear
        -- of overdue
        INSERT INTO #WatchlistStaging (AccountId, AssetClass, DaysSinceLastOverdue, EligibleForUpgrade)
        SELECT A.AccountId, A.AssetClass, A.DaysSinceLastOverdue, 'N'
        FROM PRO.LoanAccountCal A
        WHERE A.AssetClass <> 'STANDARD'
          AND A.OverdueDays = 0

        -- Rule 2: nested IF/ELSE via multi-branch CASE - the required
        -- watch period depends on how severe the asset class was
        UPDATE S
        SET S.EligibleForUpgrade = (
                CASE
                    WHEN S.DaysSinceLastOverdue IS NULL THEN 'N'
                    WHEN S.AssetClass = 'SUBSTANDARD' AND S.DaysSinceLastOverdue >= @StandardWatchPeriodDays THEN 'Y'
                    WHEN S.AssetClass IN ('DOUBTFUL', 'LOSS') AND S.DaysSinceLastOverdue >= @DoubtfulWatchPeriodDays THEN 'Y'
                    ELSE 'N'
                END
            )
        FROM #WatchlistStaging S

        -- Rule 3: sequential IF/ELSE - on the first business day of the
        -- month, apply an extra board-approval gate before anything can
        -- be marked eligible
        IF DAY(@ProcessDate) = 1
        BEGIN
            UPDATE S
            SET S.EligibleForUpgrade = 'PENDING_APPROVAL'
            FROM #WatchlistStaging S
            WHERE S.EligibleForUpgrade = 'Y'
        END
        ELSE
        BEGIN
            UPDATE S SET S.EligibleForUpgrade = S.EligibleForUpgrade
            FROM #WatchlistStaging S
        END

        -- Rule 4: upsert every staged account into the watchlist table -
        -- refresh accounts already tracked, add accounts newly clear
        MERGE PRO.NpaUpgradeWatchlist AS Target
        USING #WatchlistStaging AS Source
        ON Target.AccountId = Source.AccountId
        WHEN MATCHED THEN
            UPDATE SET
                Target.DaysSinceLastOverdue = Source.DaysSinceLastOverdue,
                Target.EligibleForUpgrade = Source.EligibleForUpgrade,
                Target.LastCheckedDate = @ProcessDate
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (AccountId, AssetClass, DaysSinceLastOverdue, EligibleForUpgrade, FirstWatchedDate, LastCheckedDate)
            VALUES (Source.AccountId, Source.AssetClass, Source.DaysSinceLastOverdue, Source.EligibleForUpgrade, @ProcessDate, @ProcessDate);

        -- Rule 5: accounts no longer clear of overdue drop off the
        -- watchlist entirely
        DELETE FROM PRO.NpaUpgradeWatchlist
        WHERE AccountId IN (
            SELECT AccountId FROM PRO.LoanAccountCal WHERE OverdueDays > 0
        )

        -- Rule 6: status transition - promote every eligible watchlist
        -- account back to Standard, unless the account has since
        -- re-classified to a different asset class than when it was
        -- staged (the join condition on AssetClass is the set-based form
        -- of that defensive consistency check - a NULL never matches, so
        -- an account with no asset class on record is excluded exactly
        -- as it would be by an explicit NULL guard)
        UPDATE A
        SET A.AssetClass = 'STANDARD',
            A.UpgradeDate = @ProcessDate
        FROM PRO.LoanAccountCal A
        INNER JOIN PRO.NpaUpgradeWatchlist W ON W.AccountId = A.AccountId AND W.AssetClass = A.AssetClass
        WHERE W.EligibleForUpgrade = 'Y'

        -- Rule 7: audit row for every promotion applied
        INSERT INTO PRO.AccountStatusAuditLog (AccountId, TransitionDate, NewStatus, Reason)
        SELECT A.AccountId, @ProcessDate, 'STANDARD', 'NPA_WATCH_PERIOD_COMPLETE'
        FROM PRO.LoanAccountCal A
        INNER JOIN PRO.NpaUpgradeWatchlist W ON W.AccountId = A.AccountId
        WHERE W.EligibleForUpgrade = 'Y'
          AND A.AssetClass = 'STANDARD'
          AND A.UpgradeDate = @ProcessDate

        DELETE FROM PRO.NpaUpgradeWatchlist
        WHERE EligibleForUpgrade = 'Y'

        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'Y', ERRORDATE = NULL, ERRORDESCRIPTION = NULL, COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'NPA_Upgrade_Watchlist_Merge'

    END TRY
    BEGIN CATCH
        -- Exception handling: record the failure for operations to investigate
        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'N', ERRORDATE = GETDATE(), ERRORDESCRIPTION = ERROR_MESSAGE(), COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'NPA_Upgrade_Watchlist_Merge'
    END CATCH
    SET NOCOUNT OFF
END
GO
