<div align="center">

<br>

# <img width="332.4" height="228.4" alt="logo" src="https://github.com/user-attachments/assets/2c4512c1-31dc-4459-b7f6-573d5f10deac" />

### OPDA: Orbital PSI Design Agent

**Human-supervised agentic AI for point-of-care design of patient-specific orbital implants**

<br>

[![Status](https://img.shields.io/badge/status-research%20prototype-F59E0B?style=for-the-badge)](#project-status)
[![Platform](https://img.shields.io/badge/platform-Windows-0078D6?style=for-the-badge&logo=windows)](#software-release)
[![Workflow](https://img.shields.io/badge/workflow-human--supervised-2563EB?style=for-the-badge)](#human-oversight)
[![Domain](https://img.shields.io/badge/domain-craniofacial%20surgery-7C3AED?style=for-the-badge)](#overview)

<br>

**[Overview](#overview) · [Architecture](#system-architecture) · [Results](#evaluation-highlights) · [Release](#software-release) · [Responsible Use](#responsible-use) · [Contact](#contact)**

<br>

> OPDA translates natural-language clinical instructions into three-dimensional orbital implant designs while keeping clinicians and biomedical engineers in control of all safety-critical decisions.

<br>

</div>

---

## Overview

**OPDA** is a human-supervised AI agent system that designs patient-specific implants for orbital fracture reconstruction. 

### 🔄 End-to-End Workflow
**Instruction** ➔ **Anatomical Reconstruction** ➔ **Orbital Prediction** ➔ **Implant Design** ➔ **Expert Review** ➔ **Manufacturing**

### ✨ Key Features
* **Hybrid Core:** Integrates multimodal foundation models, statistical shape modeling, and deterministic geometry tools.
* **Interpretable Steps:** Unlike black-box end-to-end models, OPDA decomposes tasks, verifies intermediate outputs, and utilizes dedicated tools.
* **Human-in-the-Loop:** Continuously refines and revises implant geometry based on expert feedback.

> [!IMPORTANT]
> **Research Prototype Only.** OPDA does not independently diagnose patients, determine treatment, or replace professional clinical and engineering review.

---

<div align="center">

## 📊 At a Glance

<table width="100%" cellpadding="8">
<tr>

<td align="center" valign="middle" width="18%">
<h3>🦴&nbsp;168</h3>
<sub>
<b>Printed&nbsp;assemblies</b><br>
15 clinical units
</sub>
</td>

<td align="center" valign="middle" width="18%">
<h3>⚡&nbsp;9.03&nbsp;min</h3>
<sub>
<b>Mean&nbsp;design&nbsp;time</b><br>
per implant
</sub>
</td>

<td align="center" valign="middle" width="20%">
<h3>💰&nbsp;USD&nbsp;1.91</h3>
<sub>
<b>Estimated&nbsp;design&nbsp;cost</b><br>
per implant
</sub>
</td>

<td align="center" valign="middle" width="18%">
<h3>⭐&nbsp;3.88&nbsp;/&nbsp;5</h3>
<sub>
<b>Usability&nbsp;score</b><br>
expert assessment
</sub>
</td>

<td align="center" valign="middle" width="26%">
<h3>📈&nbsp;55%&nbsp;→&nbsp;81%</h3>
<sub>
<b>First-pass&nbsp;acceptance</b><br>
after local adaptation
</sub>
</td>

</tr>
</table>

<sub><i>Results reflect the evaluated research setting.</i></sub>

</div>

---
## Why OPDA?

Traditional orbital implant design is fragmented, manual, and slow. **OPDA unifies this process into a coordinated, human-supervised AI workflow.**

<table width="100%">
<tr>
<td align="center" width="33%"><b>💬 Clinical Interaction</b><br><small>Translates natural language into structured design actions.</small></td>
<td align="center" width="34%"><b>🧠 Agentic Orchestration</b><br><small>Specialized agents plan, execute, and iterate automatically.</small></td>
<td align="center" width="33%"><b>🧑‍⚕️ Human Oversight</b><br><small>Experts retain control over design, fixation, and release.</small></td>
</tr>
</table>

<p align="center"><b>Intent</b> ➔ <b>Planning</b> ➔ <b>Execution</b> ➔ <b>Approval</b></p>

---

## System Architecture

```mermaid
flowchart TD
    A["Clinical Instruction"] --> O["OPDA Orchestrator"]

    %% 5条纵向并行的流水线 (通过 \n 换行控制方框宽度)
    O --> M["Mesh Agent"]
    M --> M1["CT Retrieval"] --> M2["Segmentation"] --> M3["Mesh\nGeneration"] --> H

    O --> S["Spatial Aligner"]
    S --> S1["Landmark\nLocalization"] --> S2["Spatial\nRegistration"] --> H

    O --> P["Mesh Predictor"]
    P --> P1["Statistical\nShape Model"] --> P2["Orbital Contour\nPrediction"] --> H

    O --> D["Design Agent"]
    D --> D1["Implant Base"] --> D2["Perforation"] --> D3["Fixation Holes"] --> D4["Geometry\nOptimization"] --> H

    O --> R["Print Agent"]
    R --> R1["Model Checking"] --> R2["Slicing"] --> R3["Manufacturing\nHandoff"] --> H

    %% 底部审核与流转
    H["Expert Review"] -->|"Revise"| O
    H -->|"Approve"| MF["Manufacturing\n"] --> SU["Surgery\n"]

    %% 样式定义
    classDef input fill:#EEF2FF,stroke:#4F46E5,stroke-width:1px,color:#111827;
    classDef agent fill:#F5F3FF,stroke:#7C3AED,stroke-width:1px,color:#111827;
    classDef tool fill:#F8FAFC,stroke:#64748B,stroke-width:1px,color:#111827;
    classDef human fill:#ECFDF5,stroke:#059669,stroke-width:2px,color:#111827;
    classDef physical fill:#FFFBEB,stroke:#D97706,stroke-width:1.5px,color:#111827;

    %% 样式应用
    class A input;
    class O,M,S,P,D,R agent;
    class M1,M2,M3,S1,S2,P1,P2,D1,D2,D3,D4,R1,R2,R3 tool;
    class H human;
    class MF,SU physical;
```

---
## Core Modules

| Module | Main responsibility |
|---|---|
| **OPDA Orchestrator** | Interprets user intent, plans the workflow, coordinates agents, and manages revision loops |
| **Mesh Agent** | Retrieves CT data, segments craniofacial anatomy, and generates standardized surface meshes |
| **Spatial Aligner** | Localizes anatomical landmarks and standardizes model position and orientation |
| **Mesh Predictor** | Estimates the intended orbital contour using statistical shape modelling |
| **Design Agent** | Generates the implant base and applies perforation, hollowing, fixation, clearance, and geometry operations |
| **Print Agent** | Checks final models, prepares manufacturing files, and supports expert-reviewed fabrication handoff |
| **Memory System** | Reuses institution-specific expert feedback without sharing patient-level data across centres |
---

## End-to-End Workflow

```text
01  Import or retrieve preoperative CT data
02  Segment the craniofacial anatomy
03  Generate and standardize the skull mesh
04  Detect landmarks and align the anatomy
05  Estimate the intended orbital contour
06  Generate the initial implant base
07  Add perforations, fixation holes, and structural constraints
08  Inspect and revise the implant geometry
09  Obtain expert approval
10  Export the final design for manufacturing review
```

### Example instruction

```text
Create a left orbital floor implant using the predicted healthy contour.
Maintain sufficient clearance from the infra-orbital rim, add clinically
appropriate perforations, and place fixation holes along the stable rim.
Return the final implant and a structured design report for expert review.
```

### Example outputs

```text
final_optimized_implant.stl
final_optimized_result.json
PSI_Final-<case>.stl
```

---

## Evaluation Highlights

OPDA was evaluated using simulated cases, printed implant–skull assemblies, multi-centre expert assessment, component-level benchmarks, and feedback-driven adaptation.

| Evaluation | Result |
|---|---:|
| Printed PSI–skull assemblies | **168** |
| Participating clinical units | **15** |
| Mean usability score | **3.88 / 5** |
| Mean design time | **9.03 min per case** |
| Estimated design cost | **USD 1.91 per case** |
| Conventional estimated labour cost | **USD 173 per case** |
| First-pass acceptance after memory adaptation | **55% → 81%** |
| EPC-Net segmentation Dice score | **0.93 ± 0.03** |
| EPC-Net HD95 | **2.62 ± 0.51 mm** |
| Internal orbital prediction mean surface distance | **0.65 mm** |

---

## Human Oversight

OPDA is designed around **mandatory expert approval**, not autonomous clinical execution.

Expert review is required for:

- segmentation quality;
- anatomical alignment;
- predicted orbital contour;
- implant coverage and geometry;
- perforation and fixation-hole placement;
- structural and manufacturing constraints;
- final release for fabrication.

```mermaid
flowchart LR
    A["AI-generated proposal"] --> B["Clinical and engineering review"]
    B -->|Approved| C["Manufacturing preparation"]
    B -->|Revision requested| D["Agentic revision loop"]
    D --> A
```

---

## Feedback-Driven Memory

OPDA can convert expert corrections into reusable local design experience.

The memory system is intended to:

- improve first-pass design acceptance;
- adapt to institution-specific design preferences;
- reduce repeated interaction and revision;
- preserve human approval at safety-critical stages;
- avoid cross-centre sharing of patient-level data;
- operate without updating foundation-model weights.

> [!NOTE]
> Memory transferability may differ across design tasks. Some preferences, particularly fixation-hole placement, can remain strongly institution-specific.

---

## Software Release

The complete OPDA software is being prepared for public release as an integrated **Windows executable installer**.

The planned release is intended to minimize environment configuration and provide a unified interface for the full workflow.

### Planned distribution

| Component | Planned availability |
|---|---|
| Windows installer | Public release |
| Example instructions and outputs | Public release |
| User and installation guides | Public release |
| Demonstration videos | Public release |
| Evaluation scripts | Where permitted |
| Patient or restricted clinical data | **Not distributed** |
| External LLM credentials | Provided by the user |
| Local-model support | Available for selected modules |

Users will need to configure their own API credentials for external large-language-model services used by selected modules. Privacy-sensitive or institution-specific functions may be operated with locally hosted models.

<details>
<summary><strong>Planned repository structure</strong></summary>

```text
OPDA/
├── README.md
├── docs/
│   ├── system_overview.md
│   ├── installation.md
│   ├── user_guide.md
│   └── responsible_use.md
├── examples/
│   ├── example_instructions/
│   └── example_outputs/
├── assets/
│   ├── figures/
│   └── videos/
├── installer/
└── LICENSE
```

</details>

---

## Project Status

**Current stage:** research prototype, manuscript review, software packaging, documentation, and release preparation.

- [x] Agentic workflow development
- [x] Simulated-case evaluation
- [x] Multi-centre printed-model assessment
- [x] Feedback-driven memory evaluation
- [ ] Public Windows installer
- [ ] Demonstration cases
- [ ] Installation and user documentation
- [ ] Additional local-model backends
- [ ] Prospective clinical evaluation
- [ ] Regulatory and quality-management preparation

---

## Data and Privacy

OPDA was developed and evaluated using retrospective, de-identified data under institutional approvals and applicable data-use conditions.

The architecture supports:

- local processing of sensitive data;
- separation of patient data from external language-model prompts;
- institution-specific memory without cross-centre patient-level sharing;
- configurable local-model deployment;
- expert review before manufacturing.

Users are responsible for compliance with local ethics approvals, data-protection regulations, institutional policies, medical-device requirements, and quality-management procedures.

---

## Responsible Use

> [!WARNING]
> OPDA is not a certified medical device and must not be used as an autonomous clinical decision or manufacturing system.

OPDA must not be used to:

- independently diagnose a patient;
- determine whether surgery is indicated;
- replace qualified surgeons or biomedical engineers;
- directly manufacture an implant without expert review;
- process patient data without appropriate authorization;
- bypass institutional validation or quality-management procedures.

Any clinical or translational use requires independent validation, risk assessment, regulatory review, and approval by the responsible institution.

---

## Citation

A formal citation will be added after publication.

```bibtex
@article{gao_opda,
  title   = {Agentic AI for point-of-care design of patient-specific orbital implants},
  author  = {Gao, Yao and others},
  journal = {Under review},
  year    = {2026}
}
```

---

## Team and Collaboration

OPDA was developed through collaboration among clinicians, biomedical engineers, computer scientists, and manufacturing partners, coordinated by the oral and maxillofacial surgery research team at **KU Leuven and University Hospitals Leuven**.

Potential collaboration areas include:

- patient-specific implant design;
- agentic AI for surgery;
- craniofacial image analysis;
- statistical shape modelling;
- medical 3D printing;
- human–AI collaboration;
- multi-centre clinical validation.

---

## Contact

<table>
<tr>
<td>

**Yao Gao**  
Department of Oral and Maxillofacial Surgery  
KU Leuven / University Hospitals Leuven  
Leuven, Belgium  

📧 `yao.gao@kuleuven.be`

</td>
</tr>
</table>

---

## License

The software license and third-party component notices will be provided with the public release.

Until then, all rights are reserved unless explicitly stated otherwise.

---

<div align="center">

<br>

### From clinical intent to manufacturing preparation

**OPDA connects clinicians, computational modelling, and medical 3D printing through a human-supervised agentic AI workflow.**

<br>

</div>
