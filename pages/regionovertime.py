import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import plotly.express as px
import plotly.graph_objects as go
import io

# Load environment variables
load_dotenv()
st.set_page_config(
    page_title="Life Expectancy and Risk Factor Decomposition Tool",
    page_icon=":world_map:",
    layout="wide"
)

# Supabase credentials
url = os.getenv("PROJECT_URL")
key = os.getenv("SECRET_PROJECT_API_KEY")

# Initialize Supabase client
supabase: Client = create_client(url, key)

# Life table calculation
def calculate_life_table(deaths, population):
    """Calculate life table from deaths and population data."""
    df = pd.DataFrame({
        'Age': ['<1 year', '12-23 months', '2-4 years', '5-9 years', '10-14 years', '15-19 years', '20-24 years',
                '25-29 years', '30-34 years', '35-39 years', '40-44 years', '45-49 years', '50-54 years',
                '55-59 years', '60-64 years', '65-69 years', '70-74 years', '75-79 years', '80-84 years',
                '85-89 years', '90-94 years', '95+ years'],
        'Years in Interval (n)': [1, 1, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
        'Deaths (nDx)': deaths,
        'Reported Population (nNx)': population
    })

    df['Mortality Rate (nmx)'] = df['Deaths (nDx)'] / df['Reported Population (nNx)']

    # Add linearly adjusted probabilities
    df['Linearity Adjustment (nax)'] = 0.5
    df.loc[0, 'Linearity Adjustment (nax)'] = 0.1
    df.loc[1, 'Linearity Adjustment (nax)'] = 0.3
    df.loc[2, 'Linearity Adjustment (nax)'] = 0.4

    df['Probability of Dying (nqx)'] = df['Years in Interval (n)'] * df['Mortality Rate (nmx)'] / \
                                       (1 + (1 - df['Linearity Adjustment (nax)']) * df['Mortality Rate (nmx)'] * df['Years in Interval (n)'])
    df.at[len(df)-1, 'Probability of Dying (nqx)'] = 1

    df['Probability of Surviving (npx)'] = 1 - df['Probability of Dying (nqx)']

    df['Individuals Surviving (lx)'] = 100000
    for i in range(1, len(df)):
        df.loc[i, 'Individuals Surviving (lx)'] = df.loc[i - 1, 'Individuals Surviving (lx)'] * df.loc[i - 1, 'Probability of Surviving (npx)']

    df['Deaths in Interval (ndx)'] = df['Individuals Surviving (lx)'] * df['Probability of Dying (nqx)']
    df.at[len(df)-1, 'Deaths in Interval (ndx)'] = df.at[len(df)-1, 'Individuals Surviving (lx)']

    df['Years Lived in Interval (nLx)'] = df['Years in Interval (n)'] * ((df['Individuals Surviving (lx)'] + df['Individuals Surviving (lx)'].shift(-1)) / 2)
    df.at[0, 'Years Lived in Interval (nLx)'] = df.at[0,'Years in Interval (n)'] * (df.at[1, 'Individuals Surviving (lx)'] + (df.at[0, 'Linearity Adjustment (nax)'] * df.at[0, 'Deaths in Interval (ndx)']))
    df.at[1, 'Years Lived in Interval (nLx)'] = df.at[1, 'Years in Interval (n)'] * (df.at[2, 'Individuals Surviving (lx)'] + (df.at[1, 'Linearity Adjustment (nax)'] * df.at[1, 'Deaths in Interval (ndx)']))
    df.at[2, 'Years Lived in Interval (nLx)'] = df.at[2, 'Years in Interval (n)'] * (df.at[3, 'Individuals Surviving (lx)'] + (df.at[2, 'Linearity Adjustment (nax)'] * df.at[2, 'Deaths in Interval (ndx)']))
    df.at[len(df)-1, 'Years Lived in Interval (nLx)'] = df.at[len(df)-1, 'Individuals Surviving (lx)'] / df.at[len(df)-1, 'Mortality Rate (nmx)']

    df['Cumulative Years Lived (Tx)'] = df['Years Lived in Interval (nLx)'][::-1].cumsum()[::-1]

    df['Expectancy of Life at Age x (ex)'] = df['Cumulative Years Lived (Tx)'] / df['Individuals Surviving (lx)']

    return df

# Decomposition calculation
def calculate_life_expectancy_contribution(life_table_1, life_table_2):
    """Calculate the contribution of each age group to life expectancy difference between two years."""
    # Ensure that the dataframes are aligned on age groups
    if not (life_table_1['Age'].equals(life_table_2['Age'])):
        raise ValueError("Age groups in the two life tables must match")

    contributions = []

    for i in range(len(life_table_1)):
        if i == len(life_table_1) - 1:  # Last age group (open-ended)
            delta_x = (life_table_1.loc[i, 'Individuals Surviving (lx)'] / life_table_1.loc[0, 'Individuals Surviving (lx)']) * \
                      (life_table_2.loc[i, 'Cumulative Years Lived (Tx)'] / life_table_2.loc[i, 'Individuals Surviving (lx)'] - \
                       life_table_1.loc[i, 'Cumulative Years Lived (Tx)'] / life_table_1.loc[i, 'Individuals Surviving (lx)'])
        else:
            # First term
            first_term = (life_table_1.loc[i, 'Individuals Surviving (lx)'] / life_table_1.loc[0, 'Individuals Surviving (lx)']) * \
                         (life_table_2.loc[i, 'Years Lived in Interval (nLx)'] / life_table_2.loc[i, 'Individuals Surviving (lx)'] - \
                          life_table_1.loc[i, 'Years Lived in Interval (nLx)'] / life_table_1.loc[i, 'Individuals Surviving (lx)'])

            # Second term
            second_term = (life_table_2.loc[i+1, 'Cumulative Years Lived (Tx)'] / life_table_1.loc[0, 'Individuals Surviving (lx)']) * \
                          ((life_table_1.loc[i, 'Individuals Surviving (lx)'] / life_table_2.loc[i, 'Individuals Surviving (lx)']) - \
                           (life_table_1.loc[i+1, 'Individuals Surviving (lx)'] / life_table_2.loc[i+1, 'Individuals Surviving (lx)']))

            delta_x = first_term + second_term

        contributions.append(delta_x)

    # Create a DataFrame for the contributions
    contribution_df = pd.DataFrame({
        'Age': life_table_1['Age'],
        'Contribution to LE difference (years)': contributions
    })

    # Add a row for the sum of contributions
    total_contribution = sum(contributions)
    total_row = pd.DataFrame({
        'Age': ['Life expectancy difference'],
        'Contribution to LE difference (years)': [total_contribution]
    })
    contribution_df = pd.concat([contribution_df, total_row], ignore_index=True)

    return contribution_df

def get_data(location_type, location_value, gender, year):
    if location_type == 'Country':
        response = supabase.table('PopulationData').select('*') \
            .eq('location_name', location_value) \
            .eq('sex_name', gender) \
            .eq('year', year) \
            .execute()
    elif location_type == 'Region':
        response = supabase.table('PopulationData').select('*') \
            .eq('region', location_value) \
            .eq('sex_name', gender) \
            .eq('year', year) \
            .execute()
    data = pd.DataFrame(response.data)
    return data

def get_deaths_and_population(data):
    # Aggregate data by age group
    aggregated_data = data.groupby('age_name').agg({
        'total_deaths': 'sum',
        'population': 'sum',
        'tobacco_deaths': 'sum',
        'alc_deaths': 'sum',
        'drug_deaths': 'sum'
    }).reset_index()

    aggregated_data = aggregated_data.set_index('age_name').reindex(age_groups).reset_index()
    aggregated_data = aggregated_data.fillna(0)
    deaths = aggregated_data['total_deaths'].tolist()
    population = aggregated_data['population'].tolist()
    # Include risk factor deaths
    tobacco_deaths = aggregated_data['tobacco_deaths'].tolist()
    alcohol_deaths = aggregated_data['alc_deaths'].tolist()
    drug_deaths = aggregated_data['drug_deaths'].tolist()
    return deaths, population, tobacco_deaths, alcohol_deaths, drug_deaths

def calculate_risk_factor_proportions_by_age(total_deaths, tobacco_deaths, alcohol_deaths, drug_deaths):
    other_deaths = [td - (tob + alc + drug) for td, tob, alc, drug in zip(total_deaths, tobacco_deaths, alcohol_deaths, drug_deaths)]

    proportions = []
    for age, td, tob, alc, drug, oth in zip(age_groups, total_deaths, tobacco_deaths, alcohol_deaths, drug_deaths, other_deaths):
        if td > 0:
            tob_prop = (tob / td)
            alc_prop = (alc / td)
            drug_prop = (drug / td)
            oth_prop = (oth / td)
        else:
            tob_prop = alc_prop = drug_prop = oth_prop = 0

        proportions.append({
            'Age': age,
            'Tobacco': tob_prop,
            'Alcohol': alc_prop,
            'Drugs': drug_prop,
            'Deaths not attributable to Tobacco, Alcohol and Drug use': oth_prop
        })

    df = pd.DataFrame(proportions)
    return df

def plot_risk_factors_by_age(df, year):
    df['Age'] = pd.Categorical(df['Age'], categories=age_groups, ordered=True)
    df = df.sort_values('Age')
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Age'],
        y=df['Tobacco'],
        name='Tobacco',
        marker_color='indianred'
    ))
    fig.add_trace(go.Bar(
        x=df['Age'],
        y=df['Alcohol'],
        name='Alcohol',
        marker_color='lightsalmon'
    ))
    fig.add_trace(go.Bar(
        x=df['Age'],
        y=df['Drugs'],
        name='Drugs',
        marker_color='gold'
    ))
    fig.add_trace(go.Bar(
        x=df['Age'],
        y=df['Deaths not attributable to Tobacco, Alcohol and Drug use'],
        name='Deaths not attributable to Tobacco, Alcohol and Drug use',
        marker_color='lightgreen'
    ))

    fig.update_layout(
        barmode='stack',
        xaxis_title='Age Group',
        yaxis_title='Proportion of Deaths',
        title=f'Risk Factor Proportions by Age Group for Year {year}',
        xaxis={'categoryorder':'array', 'categoryarray': age_groups}
    )
    return fig

def calculate_risk_factor_contributions(life_table1, life_table2, deaths1, deaths2, rf_df1, rf_df2):
    # Extract mortality rates
    mortality_rates1 = life_table1[['Age', 'Mortality Rate (nmx)']].copy()
    mortality_rates2 = life_table2[['Age', 'Mortality Rate (nmx)']].copy()
    mortality_rates1.rename(columns={'Mortality Rate (nmx)': 'Mortality Rate Year 1'}, inplace=True)
    mortality_rates2.rename(columns={'Mortality Rate (nmx)': 'Mortality Rate Year 2'}, inplace=True)

    # Prepare risk factor proportions
    rf_df1_renamed = rf_df1.copy()
    rf_df1_renamed.columns = ['Age', 'Tobacco Proportion Year 1', 'Alcohol Proportion Year 1', 'Drugs Proportion Year 1', 'Deaths not attributable to Tobacco, Alcohol and Drug use Proportion Year 1']

    rf_df2_renamed = rf_df2.copy()
    rf_df2_renamed.columns = ['Age', 'Tobacco Proportion Year 2', 'Alcohol Proportion Year 2', 'Drugs Proportion Year 2', 'Deaths not attributable to Tobacco, Alcohol and Drug use Proportion Year 2']

    # Extract delta_x
    contribution_df = calculate_life_expectancy_contribution(life_table1, life_table2)
    delta_x_df = contribution_df[contribution_df['Age'] != 'Life expectancy difference'][['Age', 'Contribution to LE difference (years)']].copy()
    delta_x_df.rename(columns={'Contribution to LE difference (years)': 'Delta e_x'}, inplace=True)

    # Merge all data
    merged_df = mortality_rates1.merge(mortality_rates2, on='Age')
    merged_df = merged_df.merge(rf_df1_renamed, on='Age')
    merged_df = merged_df.merge(rf_df2_renamed, on='Age')
    merged_df = merged_df.merge(delta_x_df, on='Age')

    # Compute risk factor contributions
    risk_factors = ['Tobacco', 'Alcohol', 'Drugs', 'Deaths not attributable to Tobacco, Alcohol and Drug use']
    contributions = []

    for index, row in merged_df.iterrows():
        mortality_rate_year1 = row['Mortality Rate Year 1']
        mortality_rate_year2 = row['Mortality Rate Year 2']
        delta_e_x = row['Delta e_x']
        age = row['Age']

        mortality_rate_diff = mortality_rate_year2 - mortality_rate_year1

        # Remove the condition that skips rows when mortality_rate_diff is zero
        for risk_factor in risk_factors:
            proportion_year1 = row[f'{risk_factor} Proportion Year 1']
            proportion_year2 = row[f'{risk_factor} Proportion Year 2']

            numerator = (mortality_rate_year2 * proportion_year2) - (mortality_rate_year1 * proportion_year1)
            denominator = mortality_rate_diff

            contribution = (numerator / denominator) * delta_e_x if denominator != 0 else 0

            contributions.append({
                'Age': age,
                'Risk Factor': risk_factor,
                'Contribution to LE difference (years)': contribution
            })

    risk_factor_contributions_df = pd.DataFrame(contributions)

    # Pivot the DataFrame
    if not risk_factor_contributions_df.empty:
        pivot_df = risk_factor_contributions_df.pivot(index='Age', columns='Risk Factor', values='Contribution to LE difference (years)').reset_index()
    else:
        # Handle empty DataFrame
        pivot_df = pd.DataFrame(columns=['Age'] + risk_factors)

    # Fill NaN values with zero
    pivot_df.fillna(0, inplace=True)

    # Calculate total contributions
    total_contributions = pivot_df[risk_factors].sum().to_frame().T
    total_contributions['Age'] = 'Total'

    # Append total contributions
    pivot_df = pd.concat([pivot_df, total_contributions], ignore_index=True)

    # Reorder columns
    pivot_df = pivot_df[['Age'] + risk_factors]

    # Set 'Age' as a categorical variable with the specified order
    pivot_df['Age'] = pd.Categorical(pivot_df['Age'], categories=age_groups + ['Total'], ordered=True)

    # Sort the DataFrame based on 'Age'
    pivot_df = pivot_df.sort_values('Age')

    return pivot_df

# Define the list of countries
countries = [
    'Australia',
    'Japan',
    'Canada',
    'United States of America',
    'Federal Republic of Germany',
    'Kingdom of the Netherlands',
    'Kingdom of Denmark',
    'French Republic',
    'Kingdom of Belgium',
    'Kingdom of Norway',
    'Kingdom of Sweden',
    'Republic of Austria',
    'Republic of Finland',
    'Republic of Korea',
    'United Kingdom of Great Britain and Northern Ireland',
    'Swiss Confederation'
]

# Get unique regions from the database
@st.cache_data
def get_regions():
    response = supabase.table('PopulationData').select('region').execute()
    data = pd.DataFrame(response.data)
    regions = data['region'].unique().tolist()
    return regions

#regions = get_regions()
regions = [
'United Kingdom',
'Australia',
'High-Income Mainland Europe',
'High-Income North America',
'High-Income East Asia',
'Nordic Countries'    
]

# Define the list of years
years = list(range(1990, 2022))  # Years from 1990 to 2021

# Define genders
genders = ['Male', 'Female']

st.sidebar.title('Filters')

# Location type selector
location_type = st.sidebar.selectbox('Select Location Type', ['Country', 'Region'])

# Based on the selection, display the appropriate options
if location_type == 'Country':
    # Country selector
    location = st.sidebar.selectbox('Select Country', countries)
else:
    # Region selector
    location = st.sidebar.selectbox('Select Region', regions)

# Gender selector
gender = st.sidebar.selectbox('Select Gender', genders)

# Year selectors
selected_years = st.sidebar.multiselect('Select Years (up to 7)', years, default=[1990, 2000, 2010, 2020])
if len(selected_years) > 7:
    st.sidebar.error("Please select up to 7 years.")

age_groups = ['<1 year', '12-23 months', '2-4 years', '5-9 years', '10-14 years', '15-19 years', '20-24 years',
              '25-29 years', '30-34 years', '35-39 years', '40-44 years', '45-49 years', '50-54 years',
              '55-59 years', '60-64 years', '65-69 years', '70-74 years', '75-79 years', '80-84 years',
              '85-89 years', '90-94 years', '95+ years']

risk_factors = ['Tobacco', 'Alcohol', 'Drugs', 'Deaths not attributable to Tobacco, Alcohol and Drug use']

# Data retrieval and processing
data_years = {}
deaths_population = {}
life_tables = {}
risk_factors_by_age = {}
for year in selected_years:
    data_years[year] = get_data(location_type, location, gender, year)
    deaths, population, tobacco_deaths, alcohol_deaths, drug_deaths = get_deaths_and_population(data_years[year])
    deaths_population[year] = {
        'deaths': deaths,
        'population': population,
        'tobacco_deaths': tobacco_deaths,
        'alcohol_deaths': alcohol_deaths,
        'drug_deaths': drug_deaths
    }
    life_tables[year] = calculate_life_table(deaths, population)
    risk_factors_by_age[year] = calculate_risk_factor_proportions_by_age(deaths, tobacco_deaths, alcohol_deaths, drug_deaths)

# Life expectancy differences and contributions
life_expectancies = {year: life_tables[year].loc[0, 'Expectancy of Life at Age x (ex)'] for year in selected_years}
le_differences = {}
contribution_dfs = {}
for i in range(len(selected_years)-1):
    year1 = selected_years[i]
    year2 = selected_years[i+1]
    le_diff = life_expectancies[year2] - life_expectancies[year1]
    le_differences[f"{year1}-{year2}"] = le_diff
    contribution_df = calculate_life_expectancy_contribution(life_tables[year1], life_tables[year2])
    contribution_dfs[f"{year1}-{year2}"] = contribution_df

# Prepare risk factor contributions
risk_factor_contributions = {}
for i in range(len(selected_years)-1):
    year1 = selected_years[i]
    year2 = selected_years[i+1]
    deaths1 = deaths_population[year1]['deaths']
    deaths2 = deaths_population[year2]['deaths']
    rf_df1 = risk_factors_by_age[year1]
    rf_df2 = risk_factors_by_age[year2]
    pivot_df = calculate_risk_factor_contributions(
        life_tables[year1],
        life_tables[year2],
        deaths1,
        deaths2,
        rf_df1,
        rf_df2
    )
    risk_factor_contributions[f"{year1}-{year2}"] = pivot_df

# Create a new DataFrame to hold the summary of contributions for each period
contribution_summary = []

for key, pivot_df in risk_factor_contributions.items():
    year1, year2 = key.split('-')

    # Total life expectancy difference for the period
    total_le_change = pivot_df.loc[pivot_df['Age'] == 'Total', risk_factors].sum(axis=1).values[0]

    # Calculate the number of years and percentage for each risk factor
    tobacco_years = pivot_df.loc[pivot_df['Age'] == 'Total', 'Tobacco'].values[0]
    alcohol_years = pivot_df.loc[pivot_df['Age'] == 'Total', 'Alcohol'].values[0]
    drugs_years = pivot_df.loc[pivot_df['Age'] == 'Total', 'Drugs'].values[0]
    other_years = pivot_df.loc[pivot_df['Age'] == 'Total', 'Deaths not attributable to Tobacco, Alcohol and Drug use'].values[0]

    # Calculate percentages
    tobacco_percentage = (tobacco_years / total_le_change) * 100 if total_le_change != 0 else 0
    alcohol_percentage = (alcohol_years / total_le_change) * 100 if total_le_change != 0 else 0
    drugs_percentage = (drugs_years / total_le_change) * 100 if total_le_change != 0 else 0
    other_percentage = (other_years / total_le_change) * 100 if total_le_change != 0 else 0

    # Append to the contribution summary
    contribution_summary.append({
        'Year Period': f'{year1}-{year2}',
        'Life Expectancy Difference': total_le_change,
        'Tobacco Years Contributed': tobacco_years,
        'Tobacco Percentage': tobacco_percentage,
        'Alcohol Years Contributed': alcohol_years,
        'Alcohol Percentage': alcohol_percentage,
        'Drugs Years Contributed': drugs_years,
        'Drugs Percentage': drugs_percentage,
        'Other Years Contributed': other_years,
        'Other Percentage': other_percentage
    })

# Convert to DataFrame
contribution_summary_df = pd.DataFrame(contribution_summary)

# Streamlit App Layout

st.title('Life Expectancy Analysis Dashboard')

st.header(f'Analysis for {location} ({location_type}) - {gender}')

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Life Tables', 'Life Expectancy Change Contribution', 'Risk Factor Proportions', 'Risk Factor Contributions to LE Change', 'Life Expectancy Contribution Summary', 'Raw Data'])
with tab1:
    st.subheader('Life Expectancy Overview')

    # Extract life expectancy at birth (age <1 year)
    life_expectancy_birth = {year: life_tables[year].loc[0, 'Expectancy of Life at Age x (ex)'] for year in selected_years}

    # Create a DataFrame for plotting
    life_expectancy_df = pd.DataFrame({
        'Year': selected_years,
        'Life Expectancy at Birth': [life_expectancy_birth[year] for year in selected_years]
    })

    # Create line chart using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=life_expectancy_df['Year'], y=life_expectancy_df['Life Expectancy at Birth'],
                             mode='lines+markers', name='Life Expectancy at Birth'))

    fig.update_layout(
        title='Life Expectancy at Birth',
        xaxis_title='Year',
        yaxis_title='Life Expectancy (years)',
        legend_title='Age'
    )

    st.plotly_chart(fig)

    # Display the life tables below the graph
    st.subheader('Life Tables')
    for year in selected_years:
        st.write(f"**Life Table for Year {year}:**")
        st.dataframe(life_tables[year])

with tab2:
    st.subheader('Life Expectancy Changes and Contributions')

    # Create two columns: one for charts (left) and one for tables (right)
    col1, col2 = st.columns([2, 1])  # Adjust the ratio as needed (2:1 for wider chart column)

    for key, contribution_df in contribution_dfs.items():
        year1, year2 = key.split('-')

        # Left column for charts
        with col1:
            st.write(f"**Contribution from {year1} to {year2}:**")
            # Exclude the total row from plotting
            contribution_plot_df = contribution_df[contribution_df['Age'] != 'Life expectancy difference']

            # Create the bar chart using Plotly Express
            fig = px.bar(
                contribution_plot_df,
                x='Age',
                y='Contribution to LE difference (years)',
                title=f'Contribution of Age Groups to Life Expectancy Change from {year1} to {year2}',
                barmode='group',
                labels={'Contribution to LE difference (years)': 'Years'}
            )
            st.plotly_chart(fig)

            # Add a divider or border between sections
            st.markdown("<hr style='border: 1px solid #ccc;'/>", unsafe_allow_html=True)

        # Right column for tables
        with col2:
            # Display the total life expectancy difference
            st.write(f"**Contribution from {year1} to {year2}:**")
            total_difference = contribution_df[contribution_df['Age'] == 'Life expectancy difference']['Contribution to LE difference (years)'].values[0]
            st.write(f"**Total Life Expectancy Difference from {year1} to {year2}:** {total_difference:.4f} years")

            # Display the DataFrame
            st.dataframe(contribution_df)
            # Add a divider or border between sections
            st.markdown("<hr style='border: 1px solid #ccc;'/>", unsafe_allow_html=True)

with tab3:
    st.subheader('Risk Factor Proportions by Age Group')

    # Create two columns: one for charts (left) and one for tables (right)
    col1, col2 = st.columns([2, 1])

    for year in selected_years:
        rf_df = risk_factors_by_age[year]

        # Left column for charts
        with col1:
            st.write(f"**Risk Factor Proportions for Year {year}:**")
            st.plotly_chart(plot_risk_factors_by_age(rf_df, year))

            # Add a divider between sections
            st.markdown("<hr style='border: 1px solid #ccc;'/>", unsafe_allow_html=True)

        # Right column for tables
        with col2:
            st.dataframe(rf_df)

            # Add a divider between sections
            st.markdown("<hr style='border: 1px solid #ccc;'/>", unsafe_allow_html=True)

with tab4:
    st.subheader('Risk Factor Contributions to Life Expectancy Change')

    # Create two columns: one for charts (left) and one for tables (right)
    col1, col2 = st.columns([2, 1])

    for key, pivot_df in risk_factor_contributions.items():
        year1, year2 = key.split('-')

        # Left column for charts
        with col1:
            st.write(f"**Risk Factor Contributions from {year1} to {year2}:**")

            # Exclude the total row for plotting
            plot_df = pivot_df[pivot_df['Age'] != 'Total']

            # Ensure 'Age' is a categorical variable ordered from youngest to oldest
            plot_df['Age'] = pd.Categorical(plot_df['Age'], categories=age_groups, ordered=True)
            plot_df = plot_df.sort_values('Age')

            # Melt the DataFrame for plotting
            plot_df_melted = plot_df.melt(id_vars='Age', value_vars=risk_factors, var_name='Risk Factor', value_name='Contribution')
            plot_df_melted = plot_df_melted.sort_values('Age')

            # Create the stacked bar chart
            fig = px.bar(
                plot_df_melted,
                x='Age',
                y='Contribution',
                barmode='group',
                color='Risk Factor',
                title=f'Risk Factor Contributions to LE Change from {year1} to {year2}',
                labels={'Contribution': 'Contribution to LE Difference (years)'},
                category_orders={'Age': age_groups}
            )
            st.plotly_chart(fig)

            # Add a divider between sections
            st.markdown("<hr style='border: 1px solid #ccc;'/>", unsafe_allow_html=True)

        # Right column for tables
        with col2:
            # Calculate total life expectancy change
            st.write(f"**Risk Factor Contributions from {year1} to {year2}:**")

            total_le_change = pivot_df.loc[pivot_df['Age'] == 'Total', risk_factors].sum(axis=1).values[0]

            st.write(f"**Total Life Expectancy Change from {year1} to {year2}: {total_le_change:.2f} years**")

            st.dataframe(pivot_df)

            # Add a divider between sections
            st.markdown("<hr style='border: 1px solid #ccc;'/>", unsafe_allow_html=True)

with tab5:
    st.subheader('Life Expectancy Contribution Summary')

    # Display the contribution summary table
    st.write("**Summary of Life Expectancy Contributions**")
    st.dataframe(contribution_summary_df)

    # Stacked bar chart for contributions
    st.write("**Stacked Bar Chart of Life Expectancy Contributions**")

    # Prepare the DataFrame for the stacked bar chart
    stacked_df = contribution_summary_df.melt(
        id_vars=['Year Period', 'Life Expectancy Difference'],
        value_vars=['Tobacco Years Contributed', 'Alcohol Years Contributed', 'Drugs Years Contributed', 'Deaths not attributable to Tobacco, Alcohol and Drug use'],
        var_name='Risk Factor',
        value_name='Years Contributed'
    )

    # Create the stacked bar chart using Plotly
    fig = px.bar(
        stacked_df,
        x='Year Period',
        y='Years Contributed',
        color='Risk Factor',
        barmode='group',
        title='Proportion of Life Expectancy Change Attributable to Each Risk Factor',
        labels={'Years Contributed': 'Years Contributed'}
    )

    st.plotly_chart(fig)

with tab6:
    st.subheader('Raw Data from Supabase')
    for year in selected_years:
        st.write(f"**Data for Year {year}:**")
        st.dataframe(data_years[year])

# Excel Download Functionality

# Create a BytesIO buffer to hold the Excel file in memory
output = io.BytesIO()

with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    # Write life tables for each year
    for year in selected_years:
        sheet_name = f'Life Table {year}'
        if len(sheet_name) > 31:
            sheet_name = f'LT {year}'
        life_tables[year].to_excel(writer, sheet_name=sheet_name, index=False)

    # Write contributions for each pair of years
    for key, contribution_df in contribution_dfs.items():
        sheet_name = f'LE Contrib {key}'
        if len(sheet_name) > 31:
            sheet_name = f'LEC {key}'
        contribution_df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Write risk factor proportions for each year
    for year in selected_years:
        sheet_name = f'Risk Factors {year}'
        if len(sheet_name) > 31:
            sheet_name = f'RF {year}'
        risk_factors_by_age[year].to_excel(writer, sheet_name=sheet_name, index=False)

    # Write risk factor contributions for each pair
    for key, pivot_df in risk_factor_contributions.items():
        sheet_name = f'RF LE Change {key}'
        if len(sheet_name) > 31:
            sheet_name = f'RFLEC {key}'
        pivot_df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Write raw data for each year
    for year in selected_years:
        sheet_name = f'Raw Data {year}'
        if len(sheet_name) > 31:
            sheet_name = f'RD {year}'
        data_years[year].to_excel(writer, sheet_name=sheet_name, index=False)

# Set the position to the beginning of the stream
output.seek(0)

# Use st.download_button to allow the user to download the Excel file
st.download_button(
    label="Download Results as Excel",
    data=output,
    file_name='analysis_results.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

