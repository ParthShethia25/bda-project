# ---------------------------------------------------------------------------
# IPL Data Analysis Project - Final Combined Script
# Author: Gemini AI (Based on user request)
# Date: April 10, 2025
# Description: Performs exploratory data analysis on IPL datasets
#              (matches.csv, deliveries.csv). Includes robust file loading.
# ---------------------------------------------------------------------------

# Phase 1: Import Libraries and Load Data
print("------------------------------------")
print("Phase 1: Loading Data and Initial Setup...")
print("------------------------------------")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os  # Essential for robust path handling

# Ignore warnings for cleaner output (optional)
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('darkgrid')
# %matplotlib inline # Magic command for Jupyter notebooks - keep commented out for standard .py scripts

# --- Robust File Path Construction ---
# Get the absolute path to the directory where the script is located
try:
    # __file__ is defined when running as a script
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive environments)
    script_dir = os.getcwd() 
    print("\nWarning: __file__ not defined. Using current working directory as script directory.")
    print(f"Make sure your CSV files are in: {script_dir}")


print(f"\nScript Directory detected as: {script_dir}")

# Construct the full paths to the CSV files relative to the script directory
deliveries_path = os.path.join(script_dir, 'deliveries.csv')
matches_path = os.path.join(script_dir, 'matches.csv')

print(f"Attempting to load deliveries from: {deliveries_path}")
print(f"Attempting to load matches from: {matches_path}")

# Load the datasets using the constructed paths
try:
    deliveries_df = pd.read_csv(deliveries_path)
    matches_df = pd.read_csv(matches_path)
    print("\nDatasets loaded successfully.")
except FileNotFoundError:
    print(f"\n--- ERROR ---")
    print(f"Could not find 'deliveries.csv' or 'matches.csv' at the expected paths.")
    print(f"Please ensure both files exist in the directory: {script_dir}")
    print(f"(Your current working directory is: {os.getcwd()})") 
    exit() # Exit the script if files are essential and not found
except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"An unexpected error occurred during file loading: {e}")
    exit()

# ---------------------------------------------------------------------------
# Phase 2: Data Exploration and Preprocessing
print("\n------------------------------------")
print("Phase 2: Data Exploration and Preprocessing...")
print("------------------------------------")

# Display basic information
print("\n--- Deliveries Data Info ---")
deliveries_df.info(memory_usage='deep') # Show memory usage too
print("\n--- Matches Data Info ---")
matches_df.info(memory_usage='deep')

# Display first few rows
print("\n--- Deliveries Data Head (First 5 Rows) ---")
print(deliveries_df.head())
print("\n--- Matches Data Head (First 5 Rows) ---")
print(matches_df.head())

# Display summary statistics for numerical columns
print("\n--- Deliveries Data Description (Numerical) ---")
print(deliveries_df.describe())
print("\n--- Matches Data Description (Numerical) ---")
print(matches_df.describe())

# Display summary statistics for object columns (like team names, cities)
print("\n--- Matches Data Description (Categorical) ---")
print(matches_df.describe(include=['object']))


# Check for Missing Values
print("\n--- Missing Values Count in Deliveries Data ---")
print(deliveries_df.isnull().sum())
print("\n--- Missing Values Count in Matches Data ---")
print(matches_df.isnull().sum())
# Note: Missing values in 'player_dismissed', 'dismissal_kind', 'fielder' are expected.
# 'umpire3' often has many missing values. 'city', 'winner' might have a few - investigate if critical.

# --- Preprocessing Steps ---
print("\n--- Performing Preprocessing Steps ---")
# Ensure 'match_id' column consistency (if 'id' exists in matches_df)
if 'id' in matches_df.columns and 'match_id' not in matches_df.columns:
    matches_df.rename(columns={'id': 'match_id'}, inplace=True)
    print("Renamed 'id' column to 'match_id' in matches_df.")

# Convert 'date' column to datetime objects
try:
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    print("Converted 'date' column to datetime objects.")
    # Extract season consistently from date, handling potential format issues
    matches_df['season'] = matches_df['date'].dt.year
    print("Extracted/Updated 'season' column from 'date'.")
except KeyError:
    print("Warning: 'date' column not found in matches_df during conversion.")
except Exception as e:
    print(f"Warning: Could not convert 'date' column or extract season: {e}")

# Optional: Merge DataFrames (can be memory intensive!)
# Consider merging only if subsequent analyses absolutely require it and memory allows.
# print("\nAttempting to merge dataframes...")
# ipl_df = pd.merge(deliveries_df, matches_df, on='match_id', how='left')
# print("DataFrames merged successfully (if uncommented).")


# ---------------------------------------------------------------------------
# Phase 3: Exploratory Data Analysis (EDA) & Visualization
print("\n------------------------------------------------")
print("Phase 3: Performing Analysis and Generating Visualizations...")
print("------------------------------------------------")

# Set default figure size for plots
plt.rcParams['figure.figsize'] = (12, 6) # Width, Height in inches

# --- Basic Match Statistics (from matches_df) ---
print("\n--- Analyzing Basic Match Statistics ---")

# 1. Number of matches per season
print("Plotting: Number of Matches Per Season...")
plt.figure(figsize=(12, 7)) # Slightly larger figure
sns.countplot(x='season', data=matches_df, palette='viridis', order = sorted(matches_df['season'].unique())) # Ensure seasons are ordered
plt.title('Number of Matches Played Per Season', fontsize=16)
plt.ylabel('Number of Matches', fontsize=12)
plt.xlabel('Season', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout() # Adjust layout to prevent overlap
plt.show()

# 2. Most frequent venues (Top 10)
print("Plotting: Top 10 Most Frequent Venues...")
plt.figure(figsize=(10, 8)) # Taller figure for vertical bars
top_venues = matches_df['venue'].value_counts().head(10)
sns.barplot(y=top_venues.index, x=top_venues.values, palette='magma', orient='h')
plt.title('Top 10 Most Frequent Venues', fontsize=16)
plt.xlabel('Number of Matches', fontsize=12)
plt.ylabel('Venue', fontsize=12)
plt.tight_layout()
plt.show()

# 3. Teams with most wins (Top 10)
print("Plotting: Top 10 Teams with Most Wins...")
# Handle cases where winner might be NaN (e.g., tied/no result matches)
plt.figure(figsize=(10, 8))
top_winners = matches_df['winner'].value_counts().dropna().head(10) # Drop NaN winners before counting
sns.barplot(y=top_winners.index, x=top_winners.values, palette='plasma', orient='h')
plt.title('Top 10 Teams with Most Wins', fontsize=16)
plt.xlabel('Number of Wins', fontsize=12)
plt.ylabel('Team', fontsize=12)
plt.tight_layout()
plt.show()

# 4. Toss Decision Impact
print("Analyzing and Plotting: Toss Decision Impact...")
toss_wins = matches_df['toss_winner'] == matches_df['winner']
plt.figure(figsize=(8, 6))
sns.countplot(x='toss_decision', data=matches_df, hue=toss_wins, palette='coolwarm')
plt.title('Toss Decision vs Match Outcome', fontsize=16)
plt.xlabel('Toss Decision', fontsize=12)
plt.ylabel('Number of Matches', fontsize=12)
# Ensure legend labels are clear
handles, _ = plt.gca().get_legend_handles_labels()
plt.legend(handles, ['Toss Winner Lost', 'Toss Winner Won'], title='Match Outcome', title_fontsize='13', fontsize='11')
plt.tight_layout()
plt.show()

toss_decision_counts = matches_df['toss_decision'].value_counts()
print(f"\nOverall Toss Decisions:\n{toss_decision_counts}")


# --- Player Performance Analysis ---
print("\n--- Analyzing Player Performance ---")

# 5. Top Run Scorers (Top 15)
print("Plotting: Top 15 Run Scorers...")
top_batsmen = deliveries_df.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False).head(15)
plt.figure(figsize=(12, 8)) # Wider figure
top_batsmen.plot(kind='bar', color=sns.color_palette('YlGnBu', 15))
plt.title('Top 15 Run Scorers in IPL History', fontsize=16)
plt.xlabel('Batsman', fontsize=12)
plt.ylabel('Total Runs Scored', fontsize=12)
plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
plt.tight_layout()
plt.show()

# 6. Most Player of the Match Awards (Top 15)
print("Plotting: Top 15 Player of the Match Winners...")
# Handle potential NaN values in player_of_match
if matches_df['player_of_match'].isnull().any():
    print(f"Note: Found {matches_df['player_of_match'].isnull().sum()} missing value(s) in 'player_of_match'. Excluding them from PoM analysis.")
    pom_counts = matches_df['player_of_match'].dropna().value_counts()
else:
    pom_counts = matches_df['player_of_match'].value_counts()

plt.figure(figsize=(10, 10)) # Make pie chart large enough
pom_counts.head(15).plot(kind='pie', autopct='%1.1f%%', startangle=140, pctdistance=0.85,
                         colors=sns.color_palette('tab20c', 15), textprops={'fontsize': 11})
plt.title('Top 15 Player of the Match Winners', fontsize=16)
plt.ylabel('') # Hide default ylabel for pie charts
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.show()

# 7. Top Wicket Takers (Top 15)
print("Plotting: Top 15 Wicket Takers...")
# Filter out non-bowler related dismissals (run out, retired hurt, etc.)
dismissal_types_for_bowler = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
wickets_df = deliveries_df[deliveries_df['dismissal_kind'].isin(dismissal_types_for_bowler)]
top_bowlers = wickets_df.groupby('bowler')['dismissal_kind'].count().sort_values(ascending=False).head(15)

plt.figure(figsize=(12, 8)) # Wider figure
top_bowlers.plot(kind='bar', color=sns.color_palette('OrRd_r', 15)) # Reversed palette
plt.title('Top 15 Wicket Takers in IPL History', fontsize=16)
plt.xlabel('Bowler', fontsize=12)
plt.ylabel('Total Wickets Taken', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- Innings Analysis (from deliveries_df) ---
print("\n--- Analyzing Innings Statistics ---")

# 8. Average Runs per Over (Across all matches)
print("Plotting: Estimated Average Runs per Over...")
# Simple estimation: mean runs per ball in that over * 6
avg_runs_per_over = deliveries_df.groupby('over')['total_runs'].mean() * 6

plt.figure(figsize=(12, 7))
avg_runs_per_over.plot(kind='line', marker='o', color='cyan', linewidth=2, markersize=8)
plt.title('Estimated Average Runs Scored per Over (Across all matches)', fontsize=16)
plt.xlabel('Over Number', fontsize=12)
plt.ylabel('Average Runs per Over (Estimated)', fontsize=12)
plt.xticks(np.arange(1, avg_runs_per_over.index.max() + 1, 1)) # Ensure all over numbers are shown as integers
plt.grid(True, which='major', linestyle='--', linewidth=0.7)
plt.ylim(bottom=max(0, avg_runs_per_over.min() - 1)) # Start y-axis near minimum value but not below 0
plt.tight_layout()
plt.show()

# 9. Distribution of Balls Bowled Per Over Number
print("Plotting: Distribution of Balls Bowled per Over...")
plt.figure(figsize=(12, 7))
balls_per_over = deliveries_df['over'].value_counts().sort_index()
balls_per_over.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Balls Bowled Per Over Number', fontsize=16)
plt.xlabel('Over Number', fontsize=12)
plt.ylabel('Number of Balls Bowled', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()

# --- Trend Analysis ---
print("\n--- Analyzing Trends Over Seasons ---")

# 10. Average Total Runs per Match Across Seasons
print("Plotting: Average Total Runs per Match Across Seasons...")
# Requires calculating total runs per match_id from deliveries_df
# then merging with matches_df to get the season.
total_runs_per_match = deliveries_df.groupby('match_id')['total_runs'].sum().reset_index()
# Merge with matches_df containing 'season' and 'match_id'
# Make sure 'season' column exists and is clean before merge
if 'season' in matches_df.columns:
    match_runs_season = pd.merge(total_runs_per_match, matches_df[['match_id', 'season']], on='match_id', how='left')

    # Check if merge was successful and season exists after merge
    if 'season' in match_runs_season.columns and not match_runs_season['season'].isnull().all():
        avg_score_per_season = match_runs_season.groupby('season')['total_runs'].mean()

        plt.figure(figsize=(12, 7))
        avg_score_per_season.plot(kind='line', marker='o', color='green', linewidth=2, markersize=8)
        plt.title('Average Total Runs per Match Across Seasons', fontsize=16)
        plt.xlabel('Season', fontsize=12)
        plt.ylabel('Average Total Runs Scored per Match', fontsize=12)
        plt.xticks(sorted(avg_score_per_season.index.unique())) # Ensure all seasons are marked
        plt.grid(True, which='both', linestyle='--', linewidth=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print("Could not perform average score per season analysis (missing 'season' column data after merge).")
else:
     print("Could not perform average score per season analysis ('season' column not found in matches_df).")


# ---------------------------------------------------------------------------
# Phase 4: Conclusion
print("\n------------------------------------")
print("Phase 4: Summary and Conclusion")
print("------------------------------------")
print("Analysis complete. Key insights derived from the plots and statistics generated:")
print("- Visualized trends in matches per season, popular venues, and top-performing teams.")
print("- Identified dominant players based on total runs, wickets taken, and Player of the Match awards.")
print("- Analyzed scoring patterns, showing run rates typically increase towards the end of innings.")
print("- Examined the relationship between toss decisions and match outcomes.")
print("- Observed potential trends in average match scores over different IPL seasons.")
print("\nFurther analysis could explore:")
print("  * Detailed player vs. player or team vs. team statistics.")
print("  * Venue-specific performance analysis (e.g., impact of toss at specific stadiums).")
print("  * Performance under pressure (e.g., run rates in chases vs. setting targets).")
print("  * Building predictive models for match outcomes or player performance (requires more advanced techniques).")

print("\n======================================")
print("=== End of IPL Data Analysis Script ===")
print("======================================")