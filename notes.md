# Project Implementation Notes: Student Dropout Early-Warning Pipeline

## 1. Data Provenance & Grant Acknowledgement
* **Dataset**: UCI Predict Students' Dropout and Academic Success (Dataset ID: 697).
* **Institutional Funding**: Supported by program SATDAP - Capacitação da Administração Pública under grant **POCI-05-5762-FSE-000191, Portugal**.
* **Language & Artifact Retentions**: 
  * The dataset originates from a Portuguese higher education institution.
  * Feature names retain Portuguese translation artifacts (e.g., `Nacionality` for Nationality).
  * Grading scales use Portuguese academic evaluation standards:
    * Entry/Admission grades: **0 to 200** scale.
    * Semester unit grades: **0 to 20** scale.

---

## 2. Leakage Boundary & Early-Warning Feature Subset
To prevent data leakage and ensure actionable early interventions, features are partitioned into two sets:
* **Early-Warning Set (29 Features)**: Includes baseline student demographics, socioeconomic factors, admission credentials, macroeconomic variables, and **1st-semester academic performance**.
* **Excluded 2nd-Semester Features (6 Features)**: 
  * `Curricular units 2nd sem (credited)`
  * `Curricular units 2nd sem (enrolled)`
  * `Curricular units 2nd sem (evaluations)`
  * `Curricular units 2nd sem (approved)`
  * `Curricular units 2nd sem (grade)`
  * `Curricular units 2nd sem (without evaluations)`
  * *Rationale*: Including 2nd-semester metrics introduces target leakage because these results are finalized after a student has already dropped out or completed the academic year.

---

## 3. Preprocessing, Encoding & Scaling Decisions

### Categorical Feature Handling
* **Encoded Nominal Features**: `Marital status`, `Application mode`, `Course`, `Previous qualification`, `Nacionality`, `Mother's qualification`, `Father's qualification`, `Mother's occupation`, `Father's occupation`.
* **Binary Flags**: `Displaced`, `Educational special needs`, `Debtor`, `Tuition fees up to date`, `Gender`, `Scholarship holder`, `International`, `Daytime/evening attendance`.
* **Encoding Strategy**:
  * Apply `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to nominal categorical IDs to prevent distance-based algorithms from interpreting arbitrary ID codes as ordinal ranks.
  * Retain binary flags as integer values `(0, 1)`.

### Numerical Feature Scaling
* **Continuous & Ratio Features**: `Age at enrollment`, `Admission grade`, `Previous qualification (grade)`, 1st-semester credit counts/grades, and macroeconomic rates (`Unemployment rate`, `Inflation rate`, `GDP`).
* **Scaling Strategy**:
  * Apply `StandardScaler()` to standardise continuous variables ($\mu = 0, \sigma = 1$) for distance-sensitive models (Logistic Regression, SVM, K-NN)[cite: 1].
  * Fit all transformers exclusively on training folds (`X_train`) inside a `scikit-learn` `ColumnTransformer` / `Pipeline` to prevent data leakage into test sets.

---

## 4. Target Formulation
* **Raw Target**: 3-class outcome (`Graduate`, `Dropout`, `Enrolled`).
* **Early-Warning Binary Formulation**:
  * **Class 1 (Positive / Target of Interest)**: `Dropout` (Student requires immediate intervention).
  * **Class 0 (Negative / Non-Dropout)**: `Graduate` + `Enrolled` (Student persists in system).