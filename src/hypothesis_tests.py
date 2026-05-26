"""
A/B Hypothesis Testing Module for Insurance Risk Analytics
Contains reusable statistical test functions
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, zscore

def chi_square_test(df, categorical_col, target_col):
    """
    Perform chi-square test for categorical variables
    
    Parameters:
    df: DataFrame
    categorical_col: column name for groups (e.g., 'Province')
    target_col: column name for binary outcome (e.g., 'HasClaim')
    
    Returns:
    chi2_stat, p_value, degrees_freedom, expected_freq
    """
    contingency_table = pd.crosstab(df[categorical_col], df[target_col])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p, dof, expected

def t_test_groups(df, group_col, group_a, group_b, value_col):
    """
    Perform two-sample t-test between two groups
    
    Parameters:
    df: DataFrame
    group_col: column containing group labels
    group_a: name of first group (control)
    group_b: name of second group (test)
    value_col: numerical column to compare
    
    Returns:
    t_stat, p_value
    """
    group_a_data = df[df[group_col] == group_a][value_col].dropna()
    group_b_data = df[df[group_col] == group_b][value_col].dropna()
    t_stat, p_value = ttest_ind(group_a_data, group_b_data, equal_var=False)
    return t_stat, p_value

def z_test_proportions(success_a, n_a, success_b, n_b):
    """
    Perform z-test for two proportions
    
    Parameters:
    success_a: number of successes in group A
    n_a: sample size of group A
    success_b: number of successes in group B
    n_b: sample size of group B
    
    Returns:
    z_stat, p_value
    """
    p1 = success_a / n_a
    p2 = success_b / n_b
    p_pooled = (success_a + success_b) / (n_a + n_b)
    
    z_stat = (p1 - p2) / np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_value

def interpret_p_value(p_value, alpha=0.05):
    """
    Interpret p-value and return decision
    """
    if p_value < alpha:
        return "REJECT H₀", f"p = {p_value:.4f} < {alpha} - Statistically significant"
    else:
        return "FAIL TO REJECT H₀", f"p = {p_value:.4f} >= {alpha} - Not statistically significant"

def calculate_claim_frequency(df, group_col):
    """
    Calculate claim frequency by group
    """
    claim_freq = df.groupby(group_col).agg(
        total_policies=('PolicyID', 'count'),
        policies_with_claims=('HasClaim', 'sum')
    )
    claim_freq['claim_frequency'] = claim_freq['policies_with_claims'] / claim_freq['total_policies']
    return claim_freq

def calculate_claim_severity(df, group_col):
    """
    Calculate average claim amount for policies WITH claims only
    """
    claims_only = df[df['TotalClaims'] > 0]
    severity = claims_only.groupby(group_col)['TotalClaims'].mean()
    return severity

def calculate_margin(df, group_col):
    """
    Calculate average margin by group
    """
    margin = df.groupby(group_col).apply(
        lambda x: (x['TotalPremium'] - x['TotalClaims']).mean()
    )
    return margin