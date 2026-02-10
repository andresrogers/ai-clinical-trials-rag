DROP TABLE IF EXISTS ai_drug_trials;

CREATE TABLE ai_drug_trials (
    nct_id VARCHAR PRIMARY KEY,
    brief_title TEXT,
    official_title TEXT,
    overall_status TEXT,
    phase TEXT,
    study_type TEXT,
    enrollment INT,
    start_date DATE,
    completion_date DATE,
    primary_completion_date DATE,
    why_stopped TEXT,
    intervention_type VARCHAR,
    intervention_name TEXT,
    intervention_other_name TEXT,
    condition_name TEXT,
    sponsor_name TEXT,
    primary_outcome TEXT,
    p_value NUMERIC,
    enrolled_seriously_affected INTEGER,
    enrolled_deaths INTEGER,
    death_percent NUMERIC,
    success_flag TEXT
);

INSERT INTO ai_drug_trials
SELECT DISTINCT ON (s.nct_id)
    s.nct_id, 
    s.brief_title, 
    s.official_title, 
    s.overall_status, 
    s.phase,
    s.study_type,
    s.enrollment::INT, 
    s.start_date::DATE, 
    s.completion_date::DATE,
    s.primary_completion_date::DATE,
    s.why_stopped,
    i.intervention_type,
    i.name as intervention_name,
    ion.name as intervention_other_name,
    c.downcase_name as condition_name, 
    sp.name as sponsor_name,
    o.title AS primary_outcome,
	oa.p_value,
    re.subjects_affected as enrolled_seriously_affected,
    ret.subjects_affected as enrolled_deaths,
    case
    	when ret.subjects_affected is null then null
    	else ret.subjects_affected::FLOAT/s.enrollment
    end as death_percent,
    CASE 
        WHEN s.overall_status = 'TERMINATED' THEN 'DEFINITE_FAIL'
        WHEN oa.p_value IS NULL THEN 'NO_RESULTS'
        WHEN oa.p_value < 0.05 THEN 'LIKELY_PASS'
        ELSE 'LIKELY_FAIL'
    END AS success_flag
FROM "Biotech".ctgov.studies s
LEFT JOIN "Biotech".ctgov.interventions i ON s.nct_id = i.nct_id
LEFT JOIN "Biotech".ctgov.intervention_other_names ion ON i.id = ion.intervention_id
LEFT JOIN "Biotech".ctgov.conditions c ON s.nct_id = c.nct_id
LEFT JOIN "Biotech".ctgov.sponsors sp ON s.nct_id = sp.nct_id
LEFT JOIN "Biotech".ctgov.outcomes o ON s.nct_id = o.nct_id AND o.outcome_type = 'PRIMARY'
LEFT JOIN "Biotech".ctgov.outcome_analyses oa ON o.id = oa.outcome_id
LEFT JOIN "Biotech".ctgov.reported_events re ON s.nct_id = re.nct_id AND re.event_type = 'serious'
LEFT JOIN "Biotech".ctgov.reported_event_totals ret ON s.nct_id = ret.nct_id AND ret.event_type = 'deaths'
WHERE s.start_date >= '2018-01-01'
and c.downcase_name ILIKE ANY (ARRAY['%lung cancer%']) --'%lymphoma%','%myeloma%','carcinoma','%leukemia%','%melanoma%']) 
AND s.phase IN ('PHASE3')
AND s.overall_status IN ('COMPLETED','ACTIVE_NOT_RECRUITING','TERMINATED')
and i.intervention_type = 'DRUG'
and s.enrollment >= 20
and s.study_type = 'INTERVENTIONAL'
;

-- Verify
SELECT COUNT(*) FROM ai_drug_trials;
