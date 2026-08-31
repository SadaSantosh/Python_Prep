"""Shared pytest fixtures for all project tests."""
import os
import sys

# Ensure all project directories are importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for subdir in [
    "project_01_student_performance_classifier",
    "project_02_corporate_employee_analytics",
    "Project_03_Telco_Customer_Churn",
    "project_04_real_estate_ai",
    "project_05_spam_phishing_detector",
]:
    path = os.path.join(PROJECT_ROOT, subdir)
    if path not in sys.path:
        sys.path.insert(0, path)
