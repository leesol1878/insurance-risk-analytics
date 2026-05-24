# Insurance Risk Analytics - AlphaCare Insurance Solutions (ACIS)

## Project Overview
This project analyzes 18 months of historical insurance claim data (Feb 2014 - Aug 2015) to optimize marketing strategy and identify low-risk targets for premium reduction.

## Business Objective
Help ACIS move from intuition-based pricing to analytics-driven decisions by:
- Understanding risk drivers across provinces, zip codes, and gender
- Building predictive models for claim severity and claim probability
- Providing data-backed recommendations for pricing optimization

## Key Metrics
- **Loss Ratio** = TotalClaims / TotalPremium
- **Margin** = TotalPremium - TotalClaims

## Project Structure

insurance-risk-analytics/
├── .github/workflows/ # CI/CD pipelines
├── data/ # Dataset (tracked with DVC)
├── notebooks/ # Jupyter notebooks for analysis
├── src/ # Reusable Python modules
├── reports/ # Final report
└── tests/ # Unit tests


## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/stsiokas/insurance-risk-analytics.git
cd insurance-risk-analytics

## Data Pipeline (DVC)

### Setup
```bash
# Install DVC
pip install dvc

# Pull data from remote
dvc pull