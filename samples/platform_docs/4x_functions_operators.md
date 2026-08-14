# 4X Platform Formula Expression Reference

## Control Flow

```
IF(Condition1...ConditionN)THEN(Condition1...ConditionN)
  (ELSEIF(Condition1...ConditionN)THEN(Condition1...ConditionN))(optional)
ELSE(Condition1...ConditionN)
```

## Functions

### Commonly used
- `ISEMPTY(<ColumnName>)`
- `MAX(<ColumnName>, [<GroupbyColumns>])`
- `COALESCE(<ColumnName/Value>,<Default value if column is null>)`

### Text
- `SUBSTR(<ColumnName>,<StartNumber>,<NumberOfCharacters>)`
- `LOWER(<ColumnName>)`
- `UPPER(<ColumnName>)`
- `LEN(<ColumnName>)`
- `CONVERT(<FieldName>,<toWhichDataType>)`
- `REGEX(<ColumnName>,<Pattern>)`
- `CONCAT(<ColumnName/Value1>,<ColumnName/Value2>...n)`
- `TRIM(<ColumnName/Value1>)`
- `REPLACE(<ColumnName/Value1>,<Value to be replaced>,<New Value>)`

### Date & time
- `SOM(<DateField>)` / `EOM(<DateField>)` — start/end of month
- `SOY(<DateField>)` / `EOY(<DateField>)` — start/end of year
- `SOFY(<DateField>)` / `EOFY(<DateField>)` — start/end of financial year
- `SOQ(<DateField>)` / `EOQ(<DateField>)` — start/end of quarter
- `DATEPART(<DateField>,<DatePartToBeExtracted>)`
- `DATEDIFF(<DateField1>,<DateField2>,<Units>)`
- `TODATE(<ValueToBeConvertedToDateFormat>)`
- `ADDDAY(<DateField1>, "Offset")`
- `PERIOD(TimeBasis, Offset, <Date>)`

### Logical
- `ISEMPTY(<ColumnName>)`
- `ISNOTEMPTY(<ColumnName>)`

### Math
- `MAX(<ColumnName>, [<GroupbyColumns>])`
- `ROUND(<ColumnName/Value>,<NoOfDecimals>)`
- `ABS(<ColumnName/Value>)`
- `FLOOR(<ColumnName/Value>,<Floor value>)`
- `CEIL(<ColumnName/Value>,<Ceil value>)`

## Operators

### Logical
- `AND(<Condition1>,<Condition2>...<ConditionN>)`
- `OR(<Condition1>,<Condition2>...<ConditionN>)`
- `NOT(<Condition>)`

### Membership / matching
- `<ColumnToBeEvaluated> IN [ListOfValues]`
- `<ColumnToBeEvaluated> NOTIN [ListOfValues]`
- `<ColumnToBeEvaluated> CONTAINS [ListOfValues]`
- `<ColumnToBeEvaluated> BEGINSWITH [ListOfValues]`
- `<ColumnToBeEvaluated> ENDSWITH [ListOfValues]`
- `<ColumnToBeEvaluated> DOESNOTCONTAINS [ListOfValues]`
- `<ColumnToBeEvaluated> HRCHYIN [ListOfValues]`
- `<ColumnToBeEvaluated> HRCHYNOTIN [ListOfValues]`

### Arithmetic
- `<NumericalField1> + <NumericalField2> +...+ <NumericalFieldN>`
- `<NumericalField1> - <NumericalField2> -...- <NumericalFieldN>`
- `<NumericalField1> / <NumericalField2>`
- `<NumericalField1> * <NumericalField2> *...* <NumericalFieldN>`

### Comparison
- `<Expression1> == <Expression2>`
- `<Expression1> != <Expression2>`
- `<Expression1> > <Expression2>`
- `<Expression1> >= <Expression2>`
- `<Expression1> < <Expression2>`
- `<Expression1> <= <Expression2>`
- `<Expression> BETWEEN [<LowerLimit>,<UpperLimit>]`

## Column references
Columns are referenced as quoted, dot-separated paths, e.g.:
`"FCT_NPA_PRODUCT"."FK_FPM"."MONITORING_PERIOD_END_DATE"`
