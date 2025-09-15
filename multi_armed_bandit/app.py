from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_widget
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from bandit import MultiArmedBandit
from htmltools import css

# Initialize the bandit and campaign data
DEFAULT_CAMPAIGNS = pd.DataFrame({
    'Campaign': ['A', 'B', 'C'],
    'CTR (%)': [2.0, 3.0, 4.0]
})

bandit = MultiArmedBandit(DEFAULT_CAMPAIGNS['Campaign'].tolist())

app_ui = ui.page_fluid(
    ui.h2("Multi-Armed Bandit Demo: Ad Campaign Optimization"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Campaign Settings"),
            ui.output_data_frame("input_df"),
            ui.input_numeric("n_impressions", "Number of Impressions to Simulate", value=100),
            ui.input_action_button("run", "Run Simulation", class_="btn-primary"),
            ui.hr(),
            ui.h4("Current Statistics"),
            ui.output_table("stats_table"),
        ),
        # Main section
        #ui.output_plot("distribution_plot"),
        output_widget("distribution_plot"),
        ui.output_plot("ctr_plot")    
    )
)

def server(input, output, session):
    # Store results
    results = reactive.Value({
        'iterations': [],
        'selected_arms': [],
        'rewards': [],
    })
    
    true_ctrs = reactive.Value({})

    @render.data_frame
    def input_df():
        """Create an editable data frame for user inputs.
        DEFAULT_CAMPAIGNS is the initial data.
        Display the data in the UI section with:
        ui.output_data_frame("input_df")

        User inputs are often processed as text.
        Access the edited data in the server section with:
        DTYPES = DEFAULT_CAMPAIGNS.dtypes
        input_df.data_view().astype(DTYPES) (optional arg selected=True if allowing for row selection)
        """
        return render.DataGrid(DEFAULT_CAMPAIGNS, editable=True)



    
    @reactive.Effect
    @reactive.event(input.run)
    def _():
        # Reset bandit
        global bandit
        campaign_df = input_df.data_view().astype(DEFAULT_CAMPAIGNS.dtypes)
        bandit = MultiArmedBandit(campaign_df['Campaign'].tolist())
        
        # Store true CTRs
        true_ctrs.set({
            row['Campaign']: row['CTR (%)'] / 100
            for _, row in campaign_df.iterrows()
        })
        
        # Pre-allocate arrays for results
        n_impressions = input.n_impressions()
        iterations = np.arange(1, n_impressions + 1)
        selected_arms = np.empty(n_impressions, dtype=object)
        rewards = np.zeros(n_impressions, dtype=bool)
        
        # Get true CTRs once
        ctrs = true_ctrs.get()
        
        # Batch size for vectorized operations
        # This balances between memory usage and computation speed
        batch_size = 1000
        
        for i in range(0, n_impressions, batch_size):
            end_idx = min(i + batch_size, n_impressions)
            batch_length = end_idx - i
            
            # Select arms for the entire batch at once
            batch_arms = [bandit.select_arm() for _ in range(batch_length)]
            selected_arms[i:end_idx] = batch_arms
            
            # Generate all rewards for the batch at once
            batch_ctrs = np.array([ctrs[arm] for arm in batch_arms])
            batch_rewards = np.random.random(batch_length) < batch_ctrs
            rewards[i:end_idx] = batch_rewards
            
            # Update bandit in batch
            for arm, reward in zip(batch_arms, batch_rewards):
                bandit.update(arm, reward)
        
        results.set({
            'iterations': iterations.tolist(),
            'selected_arms': selected_arms.tolist(),
            'rewards': rewards.tolist(),
        })

    @output
    @render.table
    def stats_table():
        if not results.get()['iterations']:
            return pd.DataFrame()
        
        stats = bandit.get_stats()
        df = pd.DataFrame([
            {
                'Campaign': name,
                'Trials': data['trials'],
                'Successes': data['successes'],
                'CTR (%)': f"{data['mean']*100:.2f}%",
                'True CTR (%)': f"{true_ctrs.get()[name]*100:.2f}%"
            }
            for name, data in stats.items()
        ])
        return df
    
    @output
    @render_widget
    def distribution_plot():
        if not results.get()['iterations']:
            return go.Figure()
        
        # Get probability distributions
        probs = bandit.get_probabilities()
        
        # Create subplot with shared x-axis
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
        
        for i, (name, samples) in enumerate(probs.items(), 1):
            fig.add_trace(
                go.Histogram(x=samples, name=name, nbinsx=30, histnorm='probability'),
                row=i, col=1
            )
        
        fig.update_layout(
            title="Probability Distributions (Beta) for Each Campaign",
            showlegend=True,
            height=600
        )
        return fig
    
    @output
    @render.plot
    def ctr_plot():
        if not results.get()['iterations']:
            return None
            
        # Convert to numpy arrays for faster computation
        iterations = np.array(results.get()['iterations'])
        campaigns = np.array(results.get()['selected_arms'])
        rewards = np.array(results.get()['rewards'])
        
        # Plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        
        # Calculate running CTRs more efficiently using numpy operations
        for campaign in bandit.arms.keys():
            mask = campaigns == campaign
            if not np.any(mask):  # Skip if campaign wasn't selected
                continue
                
            cumsum_rewards = np.cumsum(rewards[mask])
            cumsum_trials = np.cumsum(mask)
            
            # Get indices where this campaign was selected
            campaign_indices = np.where(mask)[0]
            
            plt.plot(
                iterations[campaign_indices],
                cumsum_rewards / np.arange(1, len(cumsum_rewards) + 1),
                label=f"{campaign} (True CTR: {true_ctrs.get()[campaign]*100:.1f}%)"
            )
            
        plt.xlabel('Iteration')
        plt.ylabel('Click-through Rate')
        plt.title('Running CTR by Campaign')
        plt.legend()
        plt.grid(True)

app = App(app_ui, server)
