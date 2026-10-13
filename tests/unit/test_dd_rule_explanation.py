from app.report.dd_rule_explanation import build_rule_explanation


def test_build_rule_explanation_for_simple_if():
    text = build_rule_explanation(
        "FCT_NPA_PRODUCT",
        "DPD_Overdue",
        'IF(ISNOTEMPTY("A"."X"))THEN(1)ELSE(0)',
    )
    assert text
    assert "when" in text.lower() or "if" in text.lower()
    assert "THEN(" not in text


def test_build_rule_explanation_uses_meaning_when_explainer_fails():
    text = build_rule_explanation(
        "Entity",
        "Col",
        "NOT_A_VALID_4X_FORMULA",
        business_meaning="Resets the NPA date when DPD crosses the threshold.",
    )
    assert text == "Resets the NPA date when DPD crosses the threshold."


def test_build_rule_explanation_date_clearing_and_scheme_reset():
    formula = (
        'IF(OR("AccountCal"."IntNotServicedDt" == "1900-01-01", '
        'AND("DimProduct"."SchemeType" == "ODA", "DimProduct"."Aqua_Scheme" == "Y")))'
        'THEN(NULL)ELSE("AccountCal"."IntNotServicedDt")'
    )
    text = build_rule_explanation("AccountCal", "IntNotServicedDt", formula)
    assert "NULL" in text
    assert "1900-01-01" in text
    assert "DIMPRODUCT" in text.upper() or "Aqua" in text


def test_build_rule_explanation_priority_classification_ladder():
    formula = (
        'IF(AND("AccountCal"."ASSET_NORM" == "ALWYS_NPA"))THEN(@ProcessDate)'
        'ELSEIF(AND("AccountCal"."AccountEntityID" == "PUI_CAL"."AccountEntityId"))THEN("PUI_CAL"."NPA_DATE")'
        'ELSEIF(AND("AccountCal"."UcifEntityID" == "RESTR_NPA"."UcifEntityID"))THEN("RESTR_NPA"."FinalNpaDt")'
        'ELSE("AccountCal"."FinalNpaDt")'
    )
    text = build_rule_explanation("AccountCal", "FinalNpaDt", formula)
    assert "prioritized conditions" in text.lower() or "priority" in text.lower()
    assert "PUI" in text.upper()


def test_build_rule_explanation_dpd_threshold_summary():
    formula = (
        'IF(COALESCE("AccountCal"."DPD_IntService", 0) >= '
        'COALESCE("AccountCal"."RefPeriodIntService", 0))'
        'THEN("AccountCal"."DPD_IntService")ELSE(0)'
    )
    text = build_rule_explanation("AccountCal", "DPD_IntService", formula)
    assert "days past due" in text.lower() or "DPD" in text


def test_build_rule_explanation_coalesce_fallback():
    formula = 'IF(ISEMPTY("AccountCal"."IntNotServicedDt"))THEN("AccountCal"."LastCrDate")ELSE("AccountCal"."IntNotServicedDt")'
    text = build_rule_explanation("AccountCal", "IntNotServicedDt", formula)
    assert "defaulting" in text.lower()
    assert "LastCrDate" in text


_BATCH_ONLY = "Assigns the active processing timekey / system batch execution date."


def test_build_rule_explanation_batch_timekey_parameter():
    text = build_rule_explanation("AccountCal", "TimeKey", "@TIMEKEY")
    assert text == _BATCH_ONLY
    formula = 'IF("AccountCal"."FlgDeg" == "Y")THEN(@ProcessingDate)ELSE("AccountCal"."FinalNpaDt")'
    text2 = build_rule_explanation("AccountCal", "FinalNpaDt", formula)
    assert text2 != _BATCH_ONLY
    assert "FlgDeg" in text2 or "prioritized conditions" in text2.lower()


def test_timekey_in_effective_date_filter_is_not_batch_summary():
    formula = (
        'IF(AND("AdvAcBasicDetail"."EffectiveFromTimeKey" <= @TIMEKEY))'
        'THEN("AdvAcBasicDetail"."AccountEntityId")ELSE("AccountCal"."AccountEntityId")'
    )
    text = build_rule_explanation("AccountCal", "AccountEntityId", formula)
    assert text != _BATCH_ONLY
    assert "Populates" in text or "matched on" in text.lower() or "prioritized conditions" in text.lower()


def test_build_dynamic_rule_sentence_batch_in_then_does_not_raise():
    from app.report.condition_explainer import build_dynamic_rule_sentence

    formula = 'IF("AccountCal"."FlgDeg" == "Y")THEN(@ProcessingDate)ELSE("AccountCal"."FinalNpaDt")'
    text = build_dynamic_rule_sentence(
        formula,
        target_column="FinalNpaDt",
        target_entity="AccountCal",
    )
    assert text
    assert text != _BATCH_ONLY


def test_build_rule_explanation_entity_key_passthrough():
    formula = (
        'IF(AND("AdvAcBasicDetail"."EffectiveFromTimeKey" <= @TIMEKEY))'
        'THEN("AdvAcBasicDetail"."AccountEntityId")ELSE("AccountCal"."AccountEntityId")'
    )
    text = build_rule_explanation("AccountCal", "AccountEntityId", formula)
    assert "Updates" in text or "Populates" in text
    assert "AdvAcBasicDetail" in text
    assert "matched on" in text.lower() or "AccountEntityId" in text
    assert "conditional business logic" not in text.lower()


def test_rule_explanation_has_no_markdown_bold():
    formula = (
        'IF(AND("AdvAcBasicDetail"."EffectiveFromTimeKey" <= @TIMEKEY))'
        'THEN("AdvAcBasicDetail"."AccountEntityId")ELSE("AccountCal"."AccountEntityId")'
    )
    text = build_rule_explanation("AccountCal", "AccountEntityId", formula)
    assert "**" not in text
    assert "''" not in text


def test_single_branch_effective_date_then_n_uses_sets_to_not_from():
    formula = (
        'IF(AND("AdvAcBasicDetail"."EffectiveFromTimeKey" <= @TIMEKEY))'
        'THEN("N")ELSE("AccountCal"."FlgDeg")'
    )
    text = build_rule_explanation("AccountCal", "FlgDeg", formula)
    assert " from N when" not in text
    assert "Sets FlgDeg to 'N'" in text or "to 'N'" in text


def test_nested_if_outcome_does_not_leak_ast_tree_dump():
    formula = (
        'IF("AccountCal"."Flag" == "Y")'
        'THEN(IF("AccountCal"."DPD" >= 1)THEN("AccountCal"."OverDueSinceDt")ELSE(NULL))'
        'ELSE(NULL)'
    )
    text = build_rule_explanation("AccountCal", "OverDueSinceDt", formula)
    assert "tree(" not in text.lower()
    assert "token('rule'" not in text.lower()


def test_sanitize_explanation_text_strips_ast_dumps():
    from app.report.condition_explainer import sanitize_explanation_text

    raw = "Set to tree(token('rule', 'if_expr'), [tree('compare', [])]) when matched."
    cleaned = sanitize_explanation_text(raw)
    assert "tree(" not in cleaned.lower()
    assert "calculated rule value" in cleaned.lower() or "nested" in cleaned.lower()


def test_sanitize_rewrites_updates_from_null():
    from app.report.condition_explainer import sanitize_rule_explanation_text

    raw = "Updates DebitSinceDt from NULL when FacilityType is 'BILL'; otherwise retains existing DebitSinceDt."
    cleaned = sanitize_rule_explanation_text(raw)
    assert "from NULL" not in cleaned
    assert "Resets DebitSinceDt to NULL" in cleaned


def test_build_rule_explanation_s05_dpd_max_specialized_narrator():
    formula = (
        'IF(AND(OR(COALESCE("ACCOUNTCAL"."DPD_IntService", 0) > 0, '
        'COALESCE("ACCOUNTCAL"."DPD_NoCredit", 0) > 0, '
        'COALESCE("ACCOUNTCAL"."DPD_Overdrawn", 0) > 0, '
        'COALESCE("ACCOUNTCAL"."DPD_Overdue", 0) > 0, '
        'COALESCE("ACCOUNTCAL"."DPD_Renewal", 0) > 0, '
        'COALESCE("ACCOUNTCAL"."DPD_StockStmt", 0) > 0), '
        '"CUSTOMERCAL"."FlgProcessing" == "N"))'
        'THEN("ACCOUNTCAL"."DPD_IntService")ELSE("ACCOUNTCAL"."DPD_Max")'
    )
    text = build_rule_explanation("AccountCal", "DPD_Max", formula)
    assert "prioritized conditions" in text.lower()
    assert "FlgProcessing" in text or "flgprocessing" in text.lower()
    assert "days past due" not in text.lower()


def test_build_rule_explanation_inmonthmark_processing_ladder():
    formula = (
        'IF(AND(@PROC_FREQ == "D", @PROCESSDATE != EOM(@PROCESSDATE)))'
        'THEN(IF("CUSTOMERCAL"."SourceSystemCustomerID" == "ACCOUNTCAL"."SourceSystemCustomerID")'
        'THEN("Y")ELSE("N"))ELSE("Y")'
    )
    text = build_rule_explanation("CUSTOMERCAL", "INMONTHMARK", formula)
    assert "nested conditional" not in text.lower()
    assert "Daily Non-EOM" in text
    assert "SourceSystemCustomerID" in text
    assert "Monthly / EOM" in text


def test_build_rule_explanation_completed_qualified_else_retains_existing():
    proc = "Marking_InMonthMark_Customer_Account_level"
    formula = (
        f'IF("ACLRUNNINGPROCESSSTATUS"."RUNNINGPROCESSNAME" == "{proc}")'
        'THEN("Y")ELSE("ACLRUNNINGPROCESSSTATUS"."COMPLETED")'
    )
    text = build_rule_explanation("ACLRUNNINGPROCESSSTATUS", "COMPLETED", formula)
    assert "otherwise retains existing COMPLETED" in text
    assert "sets to ACLRUNNINGPROCESSSTATUS" not in text.upper()


def test_build_rule_explanation_else_self_column_retains_existing():
    proc = "Marking_InMonthMark_Customer_Account_level"
    formula = (
        f'IF("ACLRUNNINGPROCESSSTATUS"."RUNNINGPROCESSNAME" == "{proc}")'
        'THEN("Y")ELSE("ACLRUNNINGPROCESSSTATUS"."COMPLETED")'
    )
    text = build_rule_explanation("ACLRUNNINGPROCESSSTATUS", "COMPLETED", formula)
    assert proc in text
    assert "retains existing COMPLETED" in text
    assert "sets to ACLRUNNINGPROCESSSTATUS" not in text.upper()


def test_build_rule_explanation_else_entity_only_passthrough_retains_column():
    proc = "Marking_InMonthMark_Customer_Account_level"
    formula = (
        f'IF("ACLRUNNINGPROCESSSTATUS"."RUNNINGPROCESSNAME" == "{proc}")'
        'THEN("Y")ELSE("ACLRUNNINGPROCESSSTATUS")'
    )
    text = build_rule_explanation("ACLRUNNINGPROCESSSTATUS", "COMPLETED", formula)
    assert "retains existing COMPLETED" in text
    assert "sets to 'ACLRUNNINGPROCESSSTATUS'" not in text


def test_build_rule_explanation_count_increment_retains_existing_on_entity_else():
    proc = "Marking_InMonthMark_Customer_Account_level"
    formula = (
        f'IF("ACLRUNNINGPROCESSSTATUS"."RUNNINGPROCESSNAME" == "{proc}")'
        'THEN(COUNT("ACLRUNNINGPROCESSSTATUS"."COUNT") + 1)'
        'ELSE("ACLRUNNINGPROCESSSTATUS")'
    )
    text = build_rule_explanation("ACLRUNNINGPROCESSSTATUS", "COUNT", formula)
    assert "Increments COUNT" in text
    assert "retains existing COUNT" in text


def test_build_rule_explanation_preserves_identifiers_and_y_n_literals():
    formula = (
        'IF("ACLRUNNINGPROCESSSTATUS"."RUNNINGPROCESSNAME" == '
        '"InsertDataforAssetClassficationRBL")THEN("Y")ELSE("N")'
    )
    text = build_rule_explanation("ACLRUNNINGPROCESSSTATUS", "COMPLETED", formula)
    assert "insert datafor" not in text.lower()
    assert "ACLRUNNINGPROCESSSTATUS.RUNNINGPROCESSNAME" in text.replace(" ", "")
    assert "InsertDataforAssetClassficationRBL" in text
    assert "Sets COMPLETED to 'Y'" in text
    assert "sets to 'N'" in text
    assert " from 'Y'" not in text
    assert " from N" not in text


def test_build_rule_explanation_increment_count_pattern():
    formula = (
        'IF("ACLRUNNINGPROCESSSTATUS"."COMPLETED" == "Y")'
        'THEN(COUNT("ACLRUNNINGPROCESSSTATUS"."RETRYCOUNT") + 1)ELSE(0)'
    )
    text = build_rule_explanation("ACLRUNNINGPROCESSSTATUS", "RETRYCOUNT", formula)
    assert "Increments" in text
    assert "by 1" in text
    assert "sets to 0" in text


def test_sanitize_rule_explanation_text_strips_markdown_and_double_quotes():
    from app.report.condition_explainer import sanitize_rule_explanation_text

    raw = "Updates **AccountEntityID** when ''FORMO'' is matched; sets to ''N''."
    cleaned = sanitize_rule_explanation_text(raw)
    assert cleaned == "Updates AccountEntityID when 'FORMO' is matched; sets to 'N'."


def test_single_branch_effective_date_source_column_update():
    formula = (
        'IF(AND("AdvAcRestructureDetail"."EffectiveFromTimeKey" <= @TIMEKEY, '
        '"AdvAcRestructureDetail"."EffectiveToTimeKey" >= @TIMEKEY))'
        'THEN("AdvAcRestructureDetail"."InvestmentGrade")'
        'ELSE("AccountCal"."InvestmentGrade")'
    )
    text = build_rule_explanation("AccountCal", "InvestmentGrade", formula)
    assert "conditional business logic in the source SQL" not in text
    assert "Updates" in text
    assert "AdvAcRestructureDetail" in text
    assert "InvestmentGrade" in text
    assert "matched on" in text.lower() or "active" in text.lower()
    assert "retains existing" in text.lower()


def test_single_branch_effective_date_literal_then():
    formula = (
        'IF(AND("AdvAcOtherDetail"."EffectiveFromTimeKey" <= @TIMEKEY, '
        '"AdvAcOtherDetail"."EffectiveToTimeKey" >= @TIMEKEY))'
        'THEN(0)ELSE("AccountCal"."SplCatg1Alt_Key")'
    )
    text = build_rule_explanation("AccountCal", "SplCatg1Alt_Key", formula)
    assert "conditional business logic" not in text.lower()
    assert "Sets" in text
    assert "0" in text
    assert "AdvAcOtherDetail" in text or "active" in text.lower()


def test_build_rule_explanation_splcatg_alt_key():
    formula = (
        'IF(AND("AdvAcOtherDetail"."EffectiveFromTimeKey" <= @TIMEKEY, '
        '"AdvAcOtherDetail"."EffectiveToTimeKey" >= @TIMEKEY))'
        'THEN(COALESCE("AdvAcOtherDetail"."SplCatg1Alt_Key",0))ELSE(0)'
    )
    text = build_rule_explanation("AccountCal", "SplCatg1Alt_Key", formula)
    assert "special category" in text.lower()
    assert "AdvAcOtherDetail" in text or "defaults to 0" in text.lower()


def test_build_rule_explanation_asset_class_date_reset():
    formula = (
        'IF("AccountCal"."FinalAssetClassAlt_Key" == 1)THEN(NULL)'
        'ELSE(COALESCE("AccountCal"."SrcNPA_Dt","AccountCal"."FinalNpaDt"))'
    )
    text = build_rule_explanation("AccountCal", "SrcNPA_Dt", formula)
    assert "NULL" in text
    assert "standard" in text.lower() or "historical" in text.lower()


def test_build_rule_explanation_panno_dummy_filter():
    formula = (
        'IF("CustomerCal"."PANNO" IN ["FORMO6161O","FORPM6060F","AAAAA1111A"])'
        'THEN(NULL)ELSE("CustomerCal"."PANNO")'
    )
    text = build_rule_explanation("CustomerCal", "PANNO", formula)
    assert "dummy" in text.lower() or "FORMO6161O" in text
    assert "PAN" in text.upper()
    assert "conditional business logic" not in text.lower()


def test_build_rule_explanation_membership_override_asset_norm():
    formula = (
        'IF("AccountCal"."AccountBlkCode2" IN ["K","E","W"])'
        'THEN("ALWYS_NPA")ELSE("AccountCal"."Asset_Norm")'
    )
    text = build_rule_explanation("AccountCal", "Asset_Norm", formula)
    assert "ALWYS_NPA" in text or "alwys" in text.lower()
    assert "AccountBlkCode2" in text or "blk" in text.lower()
    assert "THEN(" not in text


def test_build_rule_explanation_direct_column_passthrough():
    formula = '"CUSTOMERBASICDETAIL"."CUSTOMERNAME"'
    text = build_rule_explanation("CustomerCal", "CUSTOMERNAME", formula)
    assert "pass-through" in text.lower()
    assert "CUSTOMERBASICDETAIL" in text.upper()


def test_build_rule_explanation_avoids_generic_executive_for_nested_if():
    formula = (
        'IF(AND("DimProduct"."ProductGroup" == "LOAN"))THEN("X")'
        'ELSEIF("AccountCal"."FlgDeg" == "Y")THEN("Y")ELSE("AccountCal"."Asset_Norm")'
    )
    text = build_rule_explanation("AccountCal", "Asset_Norm", formula)
    assert "conditional business logic in the source SQL" not in text
    assert "row conditions are met" not in text.lower()
    assert "prioritized conditions" in text.lower()


def test_build_rule_explanation_nested_if_inside_coalesce():
    formula = (
        'COALESCE(IF("AccountCal"."FlgDeg" == "Y")THEN("AccountCal"."DPD_Overdue")'
        'ELSE("AccountCal"."RefPeriod"), 0)'
    )
    text = build_rule_explanation("AccountCal", "DPD_Overdue", formula)
    assert "row conditions are met" not in text.lower()
    assert "FlgDeg" in text or "prioritized conditions" in text.lower()
