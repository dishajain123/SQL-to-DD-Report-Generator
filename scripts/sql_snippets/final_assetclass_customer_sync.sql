		/* DD: consolidated customer -> account sync (3-arm precedence: Ref+Source, Source, Ucif) */
		UPDATE A
		SET
			A.FinalAssetClassAlt_Key = ISNULL(J.SysAssetClassAlt_Key, 1),
			A.FinalNpaDt = J.SysNPA_Dt
		FROM ##ACCOUNTCAL A
		INNER JOIN (
			SELECT
				A2.AccountEntityID,
				B.SysAssetClassAlt_Key,
				B.SysNPA_Dt,
				ROW_NUMBER() OVER (
					PARTITION BY A2.AccountEntityID
					ORDER BY CASE
						WHEN A2.RefCustomerID = B.RefCustomerID
							AND A2.SourceSystemCustomerID = B.SourceSystemCustomerID
							AND B.RefCustomerID <> '0' THEN 1
						WHEN A2.SourceSystemCustomerID = B.SourceSystemCustomerID THEN 2
						WHEN A2.UcifEntityID = B.UcifEntityID THEN 3
						ELSE 99
					END
				) AS SyncRank
			FROM ##ACCOUNTCAL A2
			INNER JOIN ##CustomerCal B ON (
					(A2.RefCustomerID = B.RefCustomerID AND A2.SourceSystemCustomerID = B.SourceSystemCustomerID AND B.RefCustomerID <> '0')
				OR (A2.SourceSystemCustomerID = B.SourceSystemCustomerID)
				OR (A2.UcifEntityID = B.UcifEntityID)
			)
			WHERE ISNULL(B.SysAssetClassAlt_Key, 1) <> 1
		) J ON A.AccountEntityID = J.AccountEntityID AND J.SyncRank = 1
