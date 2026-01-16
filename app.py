"""
Plateforme d'Analyse Financière - Projet Maths Finance
Auteur: Balkissa Abdourahman
Sources: PDF Projet Maths Finance, Documentation yfinance, TradingView

Références théoriques:
- Hull, J. (2018). Options, Futures, and Other Derivatives
- Tsay, R. (2010). Analysis of Financial Time Series
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# CONFIGURATION DE LA PAGE (Design professionnel)
# ============================================================================
st.set_page_config(
    page_title="ProAnalysticsSup",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SESSION STATE THEME
# ============================================================================
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# ============================================================================
# CSS DARK / LIGHT (CORRIGÉ STREAMLIT)
# ============================================================================
def inject_css():
    if st.session_state.theme == "dark":
        bg = "#0e1117"
        text = "#fafafa"
        sidebar = "#262730"
    else:
        bg = "#ffffff"
        text = "#262730"
        sidebar = "#f8f9fa"

    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-color: {bg};
        color: {text};
    }}

    [data-testid="stSidebar"] {{
        background-color: {sidebar};
    }}

    h1 {{
        color: #667eea;
        text-align: center;
    }}

    h2 {{
        color: #764ba2;
    }}

    .stButton > button {{
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }}
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ============================================================================
# BOUTON DARK / LIGHT
# ============================================================================
icon = "🌙" if st.session_state.theme == "light" else "☀️"
if st.button(icon, help="Changer le thème"):
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
    st.rerun()

# ============================================================================
# HEADER
# ============================================================================
st.markdown("<h1 style='text-align:center;'>ProAnalysticsSup</h1>", unsafe_allow_html=True)
st.divider()

# ============================================================================
# SIDEBAR - CONFIGURATION (Section 7 PDF)
# ============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Paramètres de base
    asset = st.text_input(
        " Actif financier", 
        "BTC-USD",
        help="Ex: AAPL, TSLA, BTC-USD, ETH-USD"
    )
    
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_date = st.date_input(
            " Début", 
            pd.to_datetime("2023-01-01")
        )
    with col_date2:
        end_date = st.date_input(
            " Fin", 
            pd.to_datetime("today")
        )
    
    interval = st.selectbox(
        " Fréquence",
        ["1d", "1h", "5m"],
        help="1d=Journalier, 1h=Horaire, 5m=5 minutes"
    )
    
    st.markdown("---")
    
    # Paramètres de stratégie
    st.markdown("##  Paramètres Stratégie")
    short_window = st.slider("SMA Court Terme", 5, 50, 20)
    long_window = st.slider("SMA Long Terme", 20, 200, 50)
    initial_capital = st.number_input("Capital Initial ($)", 100, 100000, 1000)
    
    st.markdown("---")
    
    # Bouton d'actualisation
    analyze_button = st.button(" Lancer l'Analyse", use_container_width=True)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def calculate_statistics(returns):
    """
    Calcul des statistiques descriptives (Section 6.2-6.3 PDF)
    
    Formules utilisées:
    - Moyenne: μ = (1/n) Σ Ri
    - Variance: σ² = (1/(n-1)) Σ (Ri - μ)²
    - Skewness: γ₁ = E[(R-μ)³]/σ³
    - Kurtosis: γ₂ = E[(R-μ)⁴]/σ⁴ - 3
    
    Source: Tsay (2010), Analysis of Financial Time Series, Chapter 2
    """
    stats_dict = {
        'mean': np.mean(returns),
        'std': np.std(returns, ddof=1),
        'skew': stats.skew(returns),
        'kurtosis': stats.kurtosis(returns),
        'min': np.min(returns),
        'max': np.max(returns),
        'median': np.median(returns)
    }
    
    # Test de normalité de Shapiro-Wilk (Section 6.3 PDF)
    # H0: Les données suivent une loi normale
    # Si p < 0.05, on rejette H0
    if len(returns) <= 5000:
        _, stats_dict['shapiro_p'] = stats.shapiro(returns)
    else:
        _, stats_dict['shapiro_p'] = stats.shapiro(returns[:5000])
    
    return stats_dict

def calculate_volatility(std, interval):
    """
    Annualisation de la volatilité (Section 6.2 PDF)
    
    Formule: σ_annuelle = σ_période × √N
    où N = nombre de périodes par an
    - Journalier: N = 252 (jours de trading)
    - Horaire: N = 252 × 6.5 (heures de trading/jour)
    - 5 minutes: N = 252 × 6.5 × 12
    
    Source: Hull (2018), Options, Futures, and Other Derivatives
    """
    periods = {
        '1d': 252,
        '1h': 252 * 6.5,
        '5m': 252 * 6.5 * 12
    }
    return std * np.sqrt(periods.get(interval, 252))

def create_candlestick_chart(data, sma_short, sma_long):
    """
    Création du graphique en chandeliers avec moyennes mobiles
    Style inspiré de TradingView (Section 3 PDF)
    """
    # Adapter le template selon le thème
    template = 'plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('Prix et Moyennes Mobiles', 'RSI (Relative Strength Index)')
    )
    
    # Chandeliers japonais
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Prix',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Moyennes mobiles
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=sma_short, 
            name=f'SMA {short_window}',
            line=dict(color='#FFA726', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=sma_long, 
            name=f'SMA {long_window}',
            line=dict(color='#42A5F5', width=2)
        ),
        row=1, col=1
    )
    
    # RSI (Relative Strength Index)
    # Formule: RSI = 100 - (100 / (1 + RS))
    # où RS = Moyenne des hausses / Moyenne des baisses
    # Source: Wilder (1978), New Concepts in Technical Trading Systems
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=rsi, 
            name='RSI',
            line=dict(color='#9C27B0', width=2)
        ),
        row=2, col=1
    )
    
    # Zones de surachat/survente
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, opacity=0.5)
    
    fig.update_layout(
        title="Analyse Technique Complète",
        xaxis_title="Date",
        yaxis_title="Prix ($)",
        height=700,
        template=template,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# ============================================================================
# ANALYSE PRINCIPALE
# ============================================================================

if analyze_button:
    with st.spinner('🔄 Téléchargement des données...'):
        try:
            # Téléchargement des données (Section 3.1 PDF - API yfinance)
            data = yf.download(asset, start=start_date, end=end_date, interval=interval, progress=False)
            
            if data.empty:
                st.error(" Aucune donnée disponible pour ces paramètres.")
            else:
                st.success(f" {len(data)} points de données chargés avec succès!")
                
                # ============================================================
                # 1. CALCUL DES RENDEMENTS (Section 6.2 PDF)
                # ============================================================
                # Formule: Rt = (Pt - Pt-1) / Pt-1 = Pt/Pt-1 - 1
                # Source: Campbell, Lo, MacKinlay (1997), The Econometrics of Financial Markets
                data['Rendement'] = data['Close'].pct_change()
                returns = data['Rendement'].dropna().values
                
                # ============================================================
                # 2. STATISTIQUES DESCRIPTIVES
                # ============================================================
                stats_results = calculate_statistics(returns)
                annual_vol = calculate_volatility(stats_results['std'], interval)
                
                # Affichage des métriques
                st.markdown("##  Statistiques Descriptives")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Rendement Moyen",
                        f"{stats_results['mean']:.4%}",
                        help="μ = (1/n) Σ Ri"
                    )
                    st.metric(
                        "Volatilité Annuelle",
                        f"{annual_vol:.4%}",
                        help="σ_ann = σ × √252"
                    )
                
                with col2:
                    st.metric(
                        "Skewness (Asymétrie)",
                        f"{stats_results['skew']:.4f}",
                        help="γ₁ = E[(R-μ)³]/σ³. >0: queue droite, <0: queue gauche"
                    )
                    st.metric(
                        "Kurtosis (Aplatissement)",
                        f"{stats_results['kurtosis']:.4f}",
                        help="γ₂ = E[(R-μ)⁴]/σ⁴ - 3. >0: queues épaisses"
                    )
                
                with col3:
                    st.metric(
                        "Rendement Min",
                        f"{stats_results['min']:.4%}"
                    )
                    st.metric(
                        "Rendement Max",
                        f"{stats_results['max']:.4%}"
                    )
                
                with col4:
                    st.metric(
                        "Médiane",
                        f"{stats_results['median']:.4%}"
                    )
                    normalite = " Normal" if stats_results['shapiro_p'] > 0.05 else " Non-normal"
                    st.metric(
                        "Test Shapiro-Wilk",
                        normalite,
                        f"p = {stats_results['shapiro_p']:.4f}",
                        help="H0: Distribution normale. Rejet si p < 0.05"
                    )
                
                # ============================================================
                # 3. GRAPHIQUE PRINCIPAL (Section 7 PDF)
                # ============================================================
                st.markdown("##  Analyse Technique")
                
                # Calcul des moyennes mobiles
                # SMA = (1/n) Σ Prix sur n périodes
                # Source: Murphy (1999), Technical Analysis of Financial Markets
                data['SMA_short'] = data['Close'].rolling(short_window).mean()
                data['SMA_long'] = data['Close'].rolling(long_window).mean()
                
                fig_main = create_candlestick_chart(data, data['SMA_short'], data['SMA_long'])
                st.plotly_chart(fig_main, use_container_width=True)
                
                # ============================================================
                # 4. BACKTESTING DE STRATÉGIE (Section 6.5 PDF)
                # ============================================================
                st.markdown("##  Backtesting - Stratégie de Croisement de Moyennes Mobiles")
                
                st.info("""
                **Règles de la stratégie:**
                -  **Signal d'achat**: SMA court terme > SMA long terme
                -  **Signal de vente**: SMA court terme < SMA long terme
                -  **Position**: 1 = Long (achat), 0 = Neutre
                
                *Source: Brock, Lakonishok & LeBaron (1992), Simple Technical Trading Rules*
                """)
                
                # Génération des signaux
                # Position(t) basée sur le signal de t-1 pour éviter le look-ahead bias
                data['Signal'] = (data['SMA_short'] > data['SMA_long']).astype(int)
                data['Position'] = data['Signal'].shift(1).fillna(0)
                
                # Calcul des rendements de la stratégie
                data['Strat_Ret'] = data['Position'] * data['Rendement']
                
                # Capital cumulatif
                # Capital(t) = Capital_initial × Π(1 + Rt)
                data['Capital_Strat'] = initial_capital * (1 + data['Strat_Ret']).cumprod()
                data['Capital_BuyHold'] = initial_capital * (1 + data['Rendement']).cumprod()
                
                # Graphique de performance
                fig_backtest = go.Figure()
                
                fig_backtest.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Capital_Strat'],
                    name='Stratégie SMA',
                    line=dict(color='#667eea', width=3)
                ))
                
                fig_backtest.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Capital_BuyHold'],
                    name='Buy & Hold',
                    line=dict(color='#f093fb', width=2, dash='dash')
                ))
                
                # Adapter le template selon le thème
                template = 'plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
                
                fig_backtest.update_layout(
                    title="Comparaison: Stratégie vs Buy & Hold",
                    xaxis_title="Date",
                    yaxis_title="Capital ($)",
                    height=500,
                    template=template,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_backtest, use_container_width=True)
                
                # Métriques de performance
                final_capital_strat = data['Capital_Strat'].iloc[-1]
                final_capital_bh = data['Capital_BuyHold'].iloc[-1]
                total_return_strat = (final_capital_strat / initial_capital - 1) * 100
                total_return_bh = (final_capital_bh / initial_capital - 1) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Capital Final (Stratégie)",
                        f"${final_capital_strat:.2f}",
                        f"{total_return_strat:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "Capital Final (Buy & Hold)",
                        f"${final_capital_bh:.2f}",
                        f"{total_return_bh:.2f}%"
                    )
                
                with col3:
                    outperformance = total_return_strat - total_return_bh
                    st.metric(
                        "Surperformance",
                        f"{outperformance:.2f}%",
                        delta_color="normal"
                    )
                
                # ============================================================
                # 5. DISTRIBUTION DES RENDEMENTS (Section 6.3 PDF)
                # ============================================================
                st.markdown("##  Analyse de Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogramme avec courbe normale
                    template = 'plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
                    
                    fig_hist = go.Figure()
                    
                    fig_hist.add_trace(go.Histogram(
                        x=returns,
                        nbinsx=50,
                        name='Rendements Observés',
                        marker_color='#667eea',
                        opacity=0.7
                    ))
                    
                    # Superposition de la loi normale théorique
                    x_range = np.linspace(returns.min(), returns.max(), 100)
                    normal_dist = stats.norm.pdf(x_range, stats_results['mean'], stats_results['std'])
                    normal_dist = normal_dist * len(returns) * (returns.max() - returns.min()) / 50
                    
                    fig_hist.add_trace(go.Scatter(
                        x=x_range,
                        y=normal_dist,
                        name='Loi Normale Théorique',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig_hist.update_layout(
                        title="Distribution des Rendements",
                        xaxis_title="Rendement",
                        yaxis_title="Fréquence",
                        height=400,
                        template=template
                    )
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # QQ-Plot (Quantile-Quantile Plot)
                    # Compare les quantiles observés aux quantiles théoriques d'une loi normale
                    # Source: Wilk & Gnanadesikan (1968), Probability plotting methods
                    fig_qq, ax = plt.subplots(figsize=(8, 6))
                    sm.qqplot(returns, line='s', ax=ax)
                    ax.set_title('QQ-Plot (Test de Normalité)', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Quantiles Théoriques', fontsize=12)
                    ax.set_ylabel('Quantiles Observés', fontsize=12)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig_qq)
                
                # ============================================================
                # 6. INTERPRÉTATION
                # ============================================================
                st.markdown("##  Interprétation des Résultats")
                
                interpretation = f"""
                ### Points Clés:
                
                1. **Distribution des rendements:**
                   - Skewness = {stats_results['skew']:.4f} → {"Distribution asymétrique à droite (gains extrêmes)" if stats_results['skew'] > 0 else "Distribution asymétrique à gauche (pertes extrêmes)"}
                   - Kurtosis = {stats_results['kurtosis']:.4f} → {"Queues épaisses (événements extrêmes fréquents)" if stats_results['kurtosis'] > 0 else "Queues fines (peu d'événements extrêmes)"}
                   - Test Shapiro: {"Hypothèse de normalité **acceptée** (p > 0.05)" if stats_results['shapiro_p'] > 0.05 else "Hypothèse de normalité **rejetée** (p < 0.05)"}
                
                2. **Performance de la stratégie:**
                   - Rendement stratégie: {total_return_strat:.2f}%
                   - Rendement Buy & Hold: {total_return_bh:.2f}%
                   - {" La stratégie surperforme" if total_return_strat > total_return_bh else " La stratégie sous-performe"}
                
                3. **Risque:**
                   - Volatilité annualisée: {annual_vol:.2%}
                   - {" Volatilité élevée - actif risqué" if annual_vol > 0.5 else " Volatilité modérée"}
                """
                
                st.markdown(interpretation)
                
                # ============================================================
                # 7. DONNÉES BRUTES
                # ============================================================
                with st.expander(" Voir les données brutes"):
                    st.dataframe(data.tail(100), use_container_width=True)
                    
                    # Option de téléchargement
                    csv = data.to_csv().encode('utf-8')
                    st.download_button(
                        label=" Télécharger les données (CSV)",
                        data=csv,
                        file_name=f'{asset}_{start_date}_{end_date}.csv',
                        mime='text/csv',
                    )
        
        except Exception as e:
            st.error(f" Erreur lors de l'analyse: {str(e)}")
            st.info(" Vérifiez le symbole de l'actif et les dates sélectionnées.")

else:
    # Message d'accueil
    st.markdown("""
    ##  Bienvenue sur ProAnalysticsSup
    
    Cette plateforme vous permet d'effectuer une **analyse quantitative complète** des marchés financiers.
    
    ###  Fonctionnalités:
    
    -  **Statistiques descriptives** (moyenne, volatilité, skewness, kurtosis)
    -  **Analyse technique** (chandeliers, moyennes mobiles, RSI)
    -  **Backtesting** de stratégies de trading
    -  **Tests de normalité** (Shapiro-Wilk, QQ-Plot)
    -  **Visualisations interactives** professionnelles
    
    ###  Pour commencer:
    
    1. Configurez vos paramètres dans le panneau latéral
    2. Cliquez sur **"Lancer l'Analyse"**
    3. Explorez les résultats et graphiques générés
    
    ---
    
    ###  Références Bibliographiques:
    
    - Hull, J. (2018). *Options, Futures, and Other Derivatives*
    - Tsay, R. (2010). *Analysis of Financial Time Series*
    - Murphy, J. (1999). *Technical Analysis of Financial Markets*
    - Campbell, Lo & MacKinlay (1997). *The Econometrics of Financial Markets*
    
    ###  Technologies Utilisées:
    
    - **Python 3.x** - Langage de programmation
    - **Streamlit** - Framework web interactif
    - **yfinance** - API de données financières (Yahoo Finance)
    - **Plotly** - Visualisations interactives
    - **SciPy/StatsModels** - Analyses statistiques
    
    ---
    
    *Développé par une stagiaire en IA  - Projet Maths Finance 2026*
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p> ProAnalysticsSupv5.0 | Développé pour un projet en analyse quantitative</p>
    <p style='font-size: 0.9rem;'>⚠️ Disclaimer: Cet outil est seulement à usage éducatif . Ne constitue pas un conseil financier.</p>
</div>
""", unsafe_allow_html=True)
