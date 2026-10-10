# Cold Chain Logistics — Sample Operating Procedures

**Document ID:** FDE-SOP-DEMO-001  
**Version:** 0.1 (training/demo)  
**Owner:** Cold Chain Operations (hypothetical)  
**Status:** SAMPLE ONLY — not an approved operating policy  
**Purpose:** Provide realistic, retrieval-friendly business instructions for the Cold Chain FDE AI agent demonstration.

> **Important:** These procedures are examples derived from the *types of signals* in the uploaded logistics dataset, not official requirements inferred from the data. Never use these instructions as real product-release, food-safety, pharmaceutical, or transport compliance guidance without an approved product-specific SOP and qualified human review.

## 1. Scope and Data Sources

The demonstration uses an operational table, `dbo.TBL_SC_FLEET_HIST_RAW`, exposed to the AI through the read-only view `FDE_VIEWS.VW_ACTIVE_FLEET`.

Relevant view fields:
- `Timestamp` — observation timestamp
- `Latitude`, `Longitude` — recorded GPS position
- `Current_Temperature_C` — IoT temperature reading in Celsius
- `Cargo_Condition_Code` — numeric cargo condition indicator; its interpretation requires an approved codebook
- `Risk_Classification` — recorded risk label (e.g., High Risk)
- `Delay_Probability` — model-derived likelihood indicator (0–1 in the sample)
- `Port_Congestion_Level` — congestion indicator; no validated threshold is provided
- `Route_Risk_Index` — route risk indicator; no validated threshold is provided

The source CSV also includes `eta_variation_hours`, `weather_condition_severity`, `driver_behavior_score`, `fatigue_monitoring_score`, and related fields, but these are **not currently exposed** in the SQL view. The AI agent must not imply it has queried these columns from this view.

**Data limitation:** The uploaded dataset does not establish product type, permitted temperature range, shipment ID, driver identity, sensor calibration status, or approved alert thresholds. Each row should be treated as an observation, not necessarily a unique shipment.

## 2. SOP — Temperature Excursion Review

**Trigger:** An observed `Current_Temperature_C` reading falls outside the *product-specific approved range*. The range must be obtained from an authorized source; **do not infer one from this dataset**.

**Procedure:**
1. Retrieve the observation time, temperature, risk classification, and available location.
2. Obtain the shipment's product-specific temperature limits from approved shipment documentation.
3. Verify the sensor reading, device calibration status, and whether the excursion is sustained or a single anomalous reading.
4. Notify the designated operations or quality contact according to the approved escalation matrix.
5. Request a review of refrigeration equipment and relevant handling records.
6. Hold potentially affected cargo from release **only under the organization’s approved quality-hold process**; a qualified person decides disposition.
7. Record observations, communications, corrective actions, and the final disposition in the incident-management system.

**Agent response rule:** Without an approved temperature range or shipment identity, state that a potential excursion cannot be confirmed. Offer the observed temperature and ask for the relevant approved range.

## 3. SOP — High-Risk Fleet Observation

**Trigger:** `Risk_Classification = 'High Risk'`.

**Procedure:**
1. Retrieve the high-risk observation with `Timestamp`, temperature, delay probability, port congestion, and route risk index.
2. Explain which observed indicators are elevated, without inventing the risk model's calculation rules.
3. Flag the record for an operations analyst to verify whether it relates to an active shipment.
4. Review shipment status, route alternatives, environmental constraints, and applicable handling requirements using authorized operational systems.
5. Escalate to the designated team when the organization's approved risk-handling criteria are met.
6. Document the review and action taken.

**Agent response rule:** The `High Risk` label is data already assigned in the dataset; do not claim that any single metric caused it unless supporting model documentation is available.

## 4. SOP — Potential Delivery Delay

**Demo trigger only:** `Delay_Probability >= 0.80` is an **illustrative training threshold**, not a company-approved threshold.

**Procedure:**
1. Identify observations matching the illustrative delay alert condition.
2. Review `Port_Congestion_Level`, `Route_Risk_Index`, and the observation timestamp.
3. Check the authoritative transportation management system for the shipment's ETA and contractual delivery commitment.
4. If the ETA is at risk under approved criteria, inform dispatch and the customer-service contact.
5. Evaluate authorized route or scheduling options and obtain approval before changes.
6. Track subsequent observations and document the final outcome.

**Agent response rule:** Describe the value as *predicted delay probability*, not a confirmed late delivery. The read-only view has no shipment ID or actual delivery status.

## 5. SOP — Elevated Route or Port Risk

**Trigger:** Route risk or port congestion indicators are flagged by an *approved, documented rule*. The uploaded data does not provide validated risk cutoffs.

**Procedure:**
1. Review the relevant risk/congestion observations and their timestamps.
2. Verify local weather, traffic, customs, and port information from approved live operational sources.
3. Assess whether the shipment has alternate permitted routes or handoff locations.
4. Seek dispatch approval for rerouting; consider refrigeration continuity and product handling constraints.
5. Record the decision and monitor for further alerts.

**Agent response rule:** Do not treat historic CSV observations as live fleet status or claim that a reroute was performed.

## 6. SOP — AI Agent Data Security and Response Rules

1. Connect using the dedicated `USR_FDE_RO` SQL account, not `sa`.
2. Only query approved semantic views such as `FDE_VIEWS.VW_ACTIVE_FLEET`.
3. Do not run INSERT, UPDATE, DELETE, DROP, ALTER, EXECUTE, or other state-changing SQL.
4. Do not expose credentials, connection strings with passwords, or internal access tokens in responses or logs.
5. Separate observed facts from recommendations, assumptions, and missing information.
6. Cite the relevant SOP title and return the observed fields used for conclusions.
7. Ask for clarification when product limits, shipment identity, or current status are absent.
8. Route operational and safety decisions to authorized humans.

## 7. Demo Questions for Retrieval Testing

- "What should an operator do if a refrigerated shipment's temperature goes outside its approved range?"
- "What steps should we follow when the database marks a fleet observation High Risk?"
- "What is the process for reviewing a high probability of delivery delay?"
- "When should route or port congestion be escalated?"
- "Can the AI agent modify fleet records?"

## 8. Example Agent Answer

**Question:** "Our data shows a High Risk record with a delay probability of 0.91. What should we do?"

**Answer:** "The record is labeled High Risk and has a predicted delay probability of 0.91. Under the sample High-Risk Observation and Potential Delivery Delay SOPs, an operations analyst should validate the current shipment status and ETA, review route and port indicators, and follow the organization's approved escalation process. This dataset alone does not confirm that a delivery is late or identify the shipment. The 0.80 delay alert cutoff used in the demo is illustrative rather than a validated operational policy."
