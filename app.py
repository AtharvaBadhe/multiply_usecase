import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from streamlit_option_menu import option_menu

# Set page configuration for a premium look
st.set_page_config(page_title="multiply.ai Strategic Command Center", layout="wide", initial_sidebar_state="expanded")

# Dark mode CSS with enhanced typography and accent colors
st.markdown("""
<style>
    .main {background-color: #1a202c;}
    .stSidebar {background-color: #2d3748; color: #e2e8f0;}
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {color: #e2e8f0;}
    .stButton>button {background-color: #4a5568; color: #e2e8f0; border-radius: 8px; border: none; padding: 8px 16px;}
    .stButton>button:hover {background-color: #5a67d8; color: white;}
    .stExpander {background-color: #2d3748; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);}
    .stExpander .stExpanderHeader {color: #e2e8f0; font-size: 18px;}
    .stTabs [data-baseweb="tab"] {background-color: #4a5568; color: #e2e8f0; border-radius: 8px; margin: 5px;}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {background-color: #5a67d8; color: white;}
    h1 {color: #5a67d8; font-family: 'Inter', sans-serif; font-weight: 700; font-size: 32px;}
    h2 {color: #e2e8f0; font-family: 'Inter', sans-serif; font-weight: 600; font-size: 24px;}
    h3 {color: #a0aec0; font-family: 'Inter', sans-serif; font-weight: 500; font-size: 20px;}
    .insight-box {background-color: #2d3748; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #5a67d8; color: #e2e8f0;}
    .pain-point {color: #ed8936; font-size: 18px; font-weight: 600;}
    .solution {color: #68d391; font-size: 18px; font-weight: 600;}
    .ceo-takeaway {color: #5a67d8; font-size: 18px; font-weight: 600;}
    .stPlotlyChart {background-color: #1a202c; border-radius: 8px; padding: 10px;}
    .stMarkdown p {color: #e2e8f0;}
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Title and Introduction
st.title("🌌 multiply.ai Strategic Command Center")
st.markdown("""
Welcome to the <b>multiply.ai Strategic Command Center</b>, a state-of-the-art dashboard designed for our CEO to drive market leadership in the UK wealth management and U.S. RIA sectors. By leveraging hiring trends, client feedback, and strategic market insights, we highlight how multiply.ai can transform operational inefficiencies into competitive advantages for firms like <b>Ascot Lloyd, Quilter, Succession Group, True Potential</b>, and U.S. RIAs.
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.header("Strategic Insights")
    selected = option_menu(
        menu_title=None,
        options=["UK Market Dynamics", "Client Voice Analysis", "U.S. RIA Conquest Plan"],
        icons=["bar-chart-line", "chat-text", "globe2"],
        default_index=0,
        styles={
            "container": {"background-color": "#2d3748"},
            "icon": {"color": "#e2e8f0", "font-size": "18px"},
            "nav-link": {"color": "#e2e8f0", "font-size": "16px", "--hover-color": "#5a67d8"},
            "nav-link-selected": {"background-color": "#5a67d8", "color": "white"},
        }
    )

# Helper Functions
def analyze_keywords(text_series, keywords_list):
    full_text = ' '.join(text_series.astype(str)).lower()
    return {keyword: full_text.count(keyword.lower()) for keyword in keywords_list}

def display_topics(model, feature_names, no_top_words):
    return [f"Topic {idx + 1}: {', '.join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])}"
            for idx, topic in enumerate(model.components_)]

# Load Data
@st.cache_data
def load_hiring_data():
    try:
        df = pd.read_csv('hiring_data_final.csv')
        df.drop(columns=['job_url', 'posted_date'], inplace=True, errors='ignore')
        df['department'].fillna('Unknown', inplace=True)
        df['high_level_department'].fillna('Unknown', inplace=True)
        df['description'].fillna('', inplace=True)
        return df
    except FileNotFoundError:
        st.error("Error: 'hiring_data_final.csv' not found.")
        return pd.DataFrame({'customer': [], 'department': [], 'description': []})

@st.cache_data
def load_reviews_data():
    try:
        df = pd.read_csv('combined_reviews_standardized.csv')
        if 'competitor' in df.columns:
            df.rename(columns={'competitor': 'customer'}, inplace=True)
        return df
    except FileNotFoundError:
        st.error("Error: 'combined_reviews_standardized.csv' not found.")
        return pd.DataFrame({'rating': [], 'review_text': [], 'customer': []})

hiring_df = load_hiring_data()
reviews_df = load_reviews_data()

# Section 1: UK Market Dynamics
if selected == "UK Market Dynamics":
    st.header("📊 Decoding UK Wealth Management Growth")
    st.markdown("""
    Uncover hiring trends for <b>Ascot Lloyd, Quilter, Succession Group, and True Potential</b> to reveal operational priorities and inefficiencies. These insights position multiply.ai to streamline critical processes and boost advisor productivity.
    """, unsafe_allow_html=True)

    # Hiring Volume
    st.subheader("Hiring Pulse: Growth Signals")
    if not hiring_df.empty:
        company_distribution = hiring_df['customer'].value_counts()
        fig = px.bar(x=company_distribution.index, y=company_distribution.values, color=company_distribution.index,
                     title="Job Postings by Firm", labels={'x': 'Firm', 'y': 'Job Postings'},
                     color_discrete_sequence=px.colors.qualitative.Dark24)
        fig.update_layout(xaxis_tickangle=45, showlegend=False, plot_bgcolor='#1a202c', paper_bgcolor='#1a202c',
                          font_color='#e2e8f0')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class='insight-box'>
        <span class='ceo-takeaway'>CEO Takeaway:</span> Elevated hiring at firms like Ascot Lloyd signals aggressive growth, straining operational systems. multiply.ai's automation can alleviate these pressures, enabling seamless scalability.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.write("No hiring data available.")

    # Firm-Specific Insights
    st.subheader("Firm-Level Operational Priorities")
    for company in ['Ascot Lloyd', 'Quilter', 'Succession Group', 'True Potential']:
        with st.expander(f"🔍 {company} Strategic Snapshot", expanded=company == 'Ascot Lloyd'):
            company_df = hiring_df[hiring_df['customer'] == company]
            if company_df.empty:
                st.write(f"No hiring data for {company}.")
            else:
                department_counts = company_df['department'].value_counts().nlargest(10)
                fig = px.bar(x=department_counts.index, y=department_counts.values, title=f"Top Hiring Departments in {company}",
                             labels={'x': 'Department', 'y': 'Job Postings'}, color_discrete_sequence=['#5a67d8'])
                fig.update_layout(xaxis_tickangle=45, plot_bgcolor='#1a202c', paper_bgcolor='#1a202c', font_color='#e2e8f0')
                st.plotly_chart(fig, use_container_width=True)

            if company == 'Ascot Lloyd':
                st.markdown("""
                <div class='insight-box'>
                <span class='pain-point'>Pain Points:</span><br>
                - ⏰ <b>High Compliance Hiring</b>: Focus on regulatory roles indicates heavy compliance workload.<br>
                - 💻 <b>IT Expansion</b>: Investment in tech roles suggests digital transformation needs.<br>
                <span class='solution'>multiply.ai Solutions:</span><br>
                - ✅ <b>Compliance Automation</b>: Reduces compliance workload by up to 30%.<br>
                - ✅ <b>AI-powered Client Portal</b>: Enhances client interactions with modern tech.<br>
                <span class='ceo-takeaway'>CEO Takeaway:</span> multiply.ai can streamline Ascot Lloyd's compliance and tech operations, freeing advisors for client-focused tasks.
                </div>
                """, unsafe_allow_html=True)
            elif company == 'Quilter':
                st.markdown("""
                <div class='insight-box'>
                <span class='pain-point'>Pain Points:</span><br>
                - ⏰ <b>Wealth Management Growth</b>: High hiring in advisory roles strains capacity.<br>
                - 🔒 <b>Cybersecurity Focus</b>: IT hiring reflects security concerns.<br>
                <span class='solution'>multiply.ai Solutions:</span><br>
                - ✅ <b>Advanced Analytics</b>: Delivers personalized advice at scale.<br>
                - ✅ <b>Enterprise-Grade Security</b>: Meets stringent cybersecurity needs.<br>
                <span class='ceo-takeaway'>CEO Takeaway:</span> multiply.ai can position Quilter as a leader in secure, client-centric wealth management.
                </div>
                """, unsafe_allow_html=True)
            elif company == 'Succession Group':
                st.markdown("""
                <div class='insight-box'>
                <span class='pain-point'>Pain Points:</span><br>
                - ⏰ <b>Operational Stability</b>: Limited hiring suggests focus on efficiency.<br>
                - 🤝 <b>Client Service Demands</b>: Need for consistent client interactions.<br>
                <span class='solution'>multiply.ai Solutions:</span><br>
                - ✅ <b>Automated Workflows</b>: Boosts efficiency without additional headcount.<br>
                - ✅ <b>Client Portal</b>: Ensures consistent client experiences.<br>
                <span class='ceo-takeaway'>CEO Takeaway:</span> multiply.ai can enhance Succession Group's efficiency and client service consistency.
                </div>
                """, unsafe_allow_html=True)
            elif company == 'True Potential':
                st.markdown("""
                <div class='insight-box'>
                <span class='pain-point'>Pain Points:</span><br>
                - 📈 <b>Advisor Growth</b>: Industry trends suggest scaling advisory teams.<br>
                - ⏰ <b>Process Scalability</b>: Expansion strains existing workflows.<br>
                <span class='solution'>multiply.ai Solutions:</span><br>
                - ✅ <b>Scalable Client Management</b>: Supports growth without quality loss.<br>
                - ✅ <b>Automation Suite</b>: Streamlines advisory processes.<br>
                <span class='ceo-takeaway'>CEO Takeaway:</span> multiply.ai can enable True Potential to scale rapidly while maintaining service excellence.
                </div>
                """, unsafe_allow_html=True)

    # Keyword Analysis
    st.subheader("Core Responsibilities & Technology Gaps")
    if not hiring_df.empty:
        responsibilities_keywords = ['client relationship', 'onboarding', 'annual review', 'financial plan', 'investment research', 'compliance', 'reporting']
        tools_keywords = ['excel', 'crm', 'database', 'software', 'platform', 'technology']
        responsibility_counts = analyze_keywords(hiring_df['description'], responsibilities_keywords)
        tool_counts = analyze_keywords(hiring_df['description'], tools_keywords)

        # Responsibilities
        resp_df = pd.DataFrame(list(responsibility_counts.items()), columns=['Responsibility', 'Frequency']).sort_values('Frequency', ascending=False)
        fig = px.bar(resp_df, x='Frequency', y='Responsibility', title='Key Responsibilities in Job Descriptions',
                     color_discrete_sequence=['#5a67d8'])
        fig.update_layout(plot_bgcolor='#1a202c', paper_bgcolor='#1a202c', font_color='#e2e8f0')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class='insight-box'>
        <span class='pain-point'>Pain Point:</span><br>
        - ⏰ <b>Time-Intensive Tasks</b>: Client relationships, onboarding, and compliance dominate advisor workloads.<br>
        <span class='solution'>multiply.ai Solution:</span><br>
        - ✅ <b>Automation Suite</b>: Reduces workload by 40% (per Kitces research), freeing advisors for high-value tasks.<br>
        <span class='ceo-takeaway'>CEO Takeaway:</span> multiply.ai can transform advisor productivity by automating critical tasks.
        </div>
        """, unsafe_allow_html=True)

        # Tools
        tools_df = pd.DataFrame(list(tool_counts.items()), columns=['Tool', 'Frequency']).sort_values('Frequency', ascending=False)
        fig = px.bar(tools_df, x='Frequency', y='Tool', title='Technology & Tools in Job Descriptions',
                     color_discrete_sequence=['#ed8936'])
        fig.update_layout(plot_bgcolor='#1a202c', paper_bgcolor='#1a202c', font_color='#e2e8f0')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class='insight-box'>
        <span class='pain-point'>Pain Point:</span><br>
        - 🔗 <b>Fragmented Tech Stack</b>: Reliance on Excel and CRM creates inefficiencies and errors.<br>
        <span class='solution'>multiply.ai Solution:</span><br>
        - ✅ <b>Integrated Platform</b>: Unifies data and automates processes for accuracy.<br>
        <span class='ceo-takeaway'>CEO Takeaway:</span> multiply.ai can eliminate tech fragmentation, boosting operational efficiency.
        </div>
        """, unsafe_allow_html=True)

# Section 2: Client Voice Analysis
elif selected == "Client Voice Analysis":
    st.header("🗣️ Harnessing Client Feedback for Impact")
    st.markdown("""
    By analyzing client reviews for <b>Ascot Lloyd, Quilter, Succession Group, and True Potential</b>, we identify operational pain points and showcase how multiply.ai can elevate client satisfaction and efficiency.
    """, unsafe_allow_html=True)

    # Review Volume
    if not reviews_df.empty:
        st.subheader("Client Engagement Snapshot")
        company_review_counts = reviews_df['customer'].value_counts()
        fig = px.pie(names=company_review_counts.index, values=company_review_counts.values,
                     title="Client Review Distribution by Firm", color_discrete_sequence=px.colors.qualitative.Dark24)
        fig.update_layout(plot_bgcolor='#1a202c', paper_bgcolor='#1a202c', font_color='#e2e8f0')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""
        <div class='insight-box'>
        <span class='pain-point'>Review Counts:</span><br>
        - True Potential: {company_review_counts.get('True Potential', 0)}<br>
        - Quilter: {company_review_counts.get('Quilter', 0)}<br>
        - Succession Group: {company_review_counts.get('Succession Group', 0)}<br>
        - Ascot Lloyd: {company_review_counts.get('Ascot Lloyd', 0)}<br>
        <span class='ceo-takeaway'>CEO Takeaway:</span> High review volumes provide a rich dataset for identifying pain points. multiply.ai can target these firms to resolve systemic issues and boost client loyalty.
        </div>
        """, unsafe_allow_html=True)

    # Sentiment Analysis
    st.subheader("Client Sentiment Landscape")
    if not reviews_df.empty:
        nltk.download('vader_lexicon', quiet=True)
        sid = SentimentIntensityAnalyzer()
        reviews_df['sentiment'] = reviews_df['review_text'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
        sentiment_summary = reviews_df.groupby('customer')['sentiment'].mean().reset_index()
        fig = px.bar(sentiment_summary, x='sentiment', y='customer', title='Average Sentiment Score by Firm',
                     color='customer', color_discrete_sequence=px.colors.qualitative.Dark24)
        fig.update_layout(plot_bgcolor='#1a202c', paper_bgcolor='#1a202c', font_color='#e2e8f0')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class='insight-box'>
        <span class='pain-point'>Pain Point:</span><br>
        - 😔 <b>Low Sentiment Scores</b>: Indicate dissatisfaction in areas like communication and service delays.<br>
        <span class='solution'>multiply.ai Solution:</span><br>
        - ✅ <b>Client Portal & AI Agents</b>: Address pain points to boost satisfaction and retention.<br>
        <span class='ceo-takeaway'>CEO Takeaway:</span> multiply.ai can transform client experiences, particularly for firms with lower sentiment scores like Quilter.
        </div>
        """, unsafe_allow_html=True)

    # Complaint Topic Modeling
    st.subheader("Root Causes of Client Frustration")
    if not reviews_df.empty:
        negative_reviews = reviews_df[reviews_df['sentiment'] < 0]['review_text']
        if not negative_reviews.empty:
            tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
            tfidf = tfidf_vectorizer.fit_transform(negative_reviews)
            nmf = NMF(n_components=4, random_state=42, l1_ratio=0.5).fit(tfidf)
            topics = display_topics(nmf, tfidf_vectorizer.get_feature_names_out(), 10)
            for topic in topics:
                st.markdown(f"- {topic}")
            st.markdown("""
            <div class='insight-box'>
            <span class='pain-point'>Pain Point:</span><br>
            - 😔 <b>Systemic Issues</b>: Communication delays, technical glitches, and process inefficiencies.<br>
            <span class='solution'>multiply.ai Solution:</span><br>
            - ✅ <b>AI Agents & Client Portal</b>: Reduce support costs and enhance client trust.<br>
            <span class='ceo-takeaway'>CEO Takeaway:</span> multiply.ai can eliminate client frustration, strengthening firm reputations.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.write("No negative reviews available for topic modeling.")

    # Firm-Specific Solutions
    st.subheader("Precision Solutions for Client Challenges")
    tabs = st.tabs(["Ascot Lloyd", "Quilter", "Succession Group", "True Potential"])
    with tabs[0]:
        st.markdown("""
        <div class='insight-box'>
        <span class='pain-point'>Pain Points:</span><br>
        - 😔 <b>Communication Delays</b>: Unreturned emails and slow responses.<br>
        - 🛠️ <b>Administrative Errors</b>: Inefficiencies in data and document processing.<br>
        <span class='solution'>multiply.ai Solutions:</span><br>
        - ✅ <b>AI-powered Client Portal</b>: Provides 24/7 access to updates.<br>
        - ✅ <b>Automation Suite</b>: Reduces errors in data processing.<br>
        - ✅ <b>Advice Engine</b>: Speeds up query resolution.<br>
        <span class='ceo-takeaway'>CEO Takeaway:</span> multiply.ai can transform Ascot Lloyd's client interactions, boosting satisfaction and efficiency.
        </div>
        """, unsafe_allow_html=True)
    with tabs[1]:
        st.markdown("""
        <div class='insight-box'>
        <span class='pain-point'>Pain Points:</span><br>
        - ⏰ <b>Transaction Delays</b>: Slow pension and withdrawal processes.<br>
        - 😔 <b>Communication Breakdowns</b>: Inconsistent client interactions.<br>
        <span class='solution'>multiply.ai Solutions:</span><br>
        - ✅ <b>Advice Engine</b>: Streamlines transaction workflows.<br>
        - ✅ <b>AI Agents</b>: Offers instant query resolution.<br>
        - ✅ <b>Client Portal</b>: Enhances transparency and satisfaction.<br>
        <span class='ceo-takeaway'>CEO Takeaway:</span> multiply.ai can position Quilter as a leader in client-centric digital wealth management.
        </div>
        """, unsafe_allow_html=True)
    with tabs[2]:
        st.markdown("""
        <div class='insight-box'>
        <span class='pain-point'>Pain Points:</span><br>
        - ⏰ <b>Advisor Availability</b>: Limited time for client interactions.<br>
        - 😔 <b>Service Inconsistency</b>: Variable quality across advisors.<br>
        <span class='solution'>multiply.ai Solutions:</span><br>
        - ✅ <b>Automated Annual Reviews</b>: Frees advisors for client engagement.<br>
        - ✅ <b>Standardized Workflows</b>: Ensures consistent service.<br>
        <span class='ceo-takeaway'>CEO Takeaway:</span> multiply.ai can enhance Succession Group's client relationships by optimizing advisor efficiency.
        </div>
        """, unsafe_allow_html=True)
    with tabs[3]:
        st.markdown("""
        <div class='insight-box'>
        <span class='pain-point'>Pain Points:</span><br>
        - 🛠️ <b>Technical/App Issues</b>: Unreliable digital interfaces.<br>
        - ⏰ <b>Complex Query Delays</b>: Slow resolution of technical queries.<br>
        <span class='solution'>multiply.ai Solutions:</span><br>
        - ✅ <b>Robust Client Portal</b>: Minimizes technical issues.<br>
        - ✅ <b>AI Agents</b>: Resolves queries instantly or routes efficiently.<br>
        <span class='ceo-takeaway'>CEO Takeaway:</span> multiply.ai can solidify True Potential's reputation with a flawless digital experience.
        </div>
        """, unsafe_allow_html=True)

# Section 3: U.S. RIA Conquest Plan
elif selected == "U.S. RIA Conquest Plan":
    st.header("🌎 Strategic Blueprint for U.S. RIA Dominance")
    st.markdown("""
    This section unveils a precision-engineered go-to-market (GTM) strategy to conquer the U.S. RIA market, targeting <b>Scaling-Constrained Practitioners (1-5 advisors)</b> and <b>Efficiency-Seeking Mid-Sized RIAs (6-50 advisors)</b>. multiply.ai's solutions address their critical pain points for rapid market penetration.
    """, unsafe_allow_html=True)

    # 40% Time Loss Visualization
    st.subheader("Critical RIA Challenge: Time Loss")
    fig = go.Figure(data=[
        go.Pie(labels=['Automatable Tasks', 'Revenue-Generating Tasks'], values=[40, 60], hole=0.4,
               marker_colors=['#ed8936', '#5a67d8'])
    ])
    fig.update_layout(title="40% of Advisor Time Lost to Automatable Tasks",
                      plot_bgcolor='#1a202c', paper_bgcolor='#1a202c', font_color='#e2e8f0')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class='insight-box'>
    <span class='ceo-takeaway'>CEO Takeaway:</span> Advisors lose 40% of their time to automatable tasks (Kitces, FPA research). multiply.ai's <b>Automation Suite</b> can reclaim this time, driving significant ROI.
    </div>
    """, unsafe_allow_html=True)

    # Market Segments
    st.subheader("Target Segments: Pain Points & Solutions")
    with st.expander("Segment Alpha: Scaling-Constrained Practitioners (1-5 Advisors)", expanded=True):
        st.markdown("""
        <div class='insight-box'>
        <span class='pain-point'>Pain Points (Kitces, FPA):</span><br>
        - ⏰ <b>Administrative Overload</b>: 40% of time on compliance, meeting prep.<br>
        - 📉 <b>Inability to Scale</b>: Growth limited by hiring and capital constraints.<br>
        - 🏆 <b>Competitive Vulnerability</b>: Outpaced by larger firms' digital experiences.<br>
        <span class='solution'>multiply.ai Solutions:</span><br>
        - ✅ <b>Automation Suite</b>: Frees advisors from administrative tasks.<br>
        - ✅ <b>Scalable Processes</b>: Supports growth without additional hires.<br>
        - ✅ <b>Client Portal</b>: Delivers competitive digital experiences.<br>
        <span class='ceo-takeaway'>CEO Takeaway:</span> multiply.ai can empower small RIAs to scale rapidly, capturing a significant share of this large segment.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Segment Bravo: Efficiency-Seeking Mid-Sized RIAs (6-50 Advisors)", expanded=True):
        st.markdown("""
        <div class='insight-box'>
        <span class='pain-point'>Pain Points (Ezra Group, WealthManagement.com):</span><br>
        - 🔗 <b>Technology Fragmentation</b>: 5-10 disjointed apps create data silos.<br>
        - 🛡️ <b>Process & Compliance Risk</b>: Inconsistent workflows elevate risks.<br>
        - 😔 <b>Inconsistent Client Experience</b>: Variable service quality across advisors.<br>
        <span class='solution'>multiply.ai Solutions:</span><br>
        - ✅ <b>CRM Integrations</b>: Unifies Salesforce, Redtail, Wealthbox.<br>
        - ✅ <b>Standardized Workflows</b>: Ensures compliance and consistency.<br>
        - ✅ <b>Client Portal</b>: Delivers uniform client experiences.<br>
        <span class='ceo-takeaway'>CEO Takeaway:</span> multiply.ai can streamline mid-sized RIAs' operations, establishing it as an essential platform.
        </div>
        """, unsafe_allow_html=True)

    # Procurement Process
    st.subheader("Navigating Procurement: Tailored Strategies")
    st.markdown("""
    <div class='insight-box'>
        <span class='pain-point'>Segment Alpha:</span><br>
        - ⏰ <b>Procurement Pain</b>: Founder-driven, short sales cycle (<30 days), price-sensitive, demands free trial.<br>
        <span class='pain-point'>Segment Bravo:</span><br>
        - ⏰ <b>Procurement Pain</b>: Decision-Making Unit (CEO, COO, CTO, CCO), long sales cycle (3-9 months), requires demos, security reviews, ROI analysis.<br>
        <span class='solution'>multiply.ai Solution:</span><br>
        - ✅ <b>Dual Sales Strategy</b>: Low-friction for small firms, high-touch for mid-sized RIAs.<br>
        <span class='ceo-takeaway'>CEO Takeaway:</span> Tailored sales approaches maximize conversions across both segments.
    </div>
    """, unsafe_allow_html=True)

    # Go-to-Market Strategy
    st.subheader("GTM Roadmap: Securing Market Leadership")
    st.markdown("""
    <div class='insight-box'>
        <span class='solution'>Segment Alpha Strategy:</span><br>
        - ✅ <b>Content & SEO</b>: Resources like "Automating RIA Quarterly Reviews" to attract advisors.<br>
        - ✅ <b>Ecosystem Marketing</b>: Integrate with Redtail, Wealthbox, Orion.<br>
        - ✅ <b>Podcast Sponsorships</b>: Partner with The Michael Kitces Podcast.<br>
        <span class='solution'>Segment Bravo Strategy:</span><br>
        - ✅ <b>Account-Based Marketing (ABM)</b>: Target 100 firms with personalized outreach.<br>
        - ✅ <b>Conferences</b>: Secure meetings at T3 Advisor Conference, Schwab IMPACT.<br>
        - ✅ <b>Case Studies</b>: Highlight metrics (e.g., "20 hours/month saved on compliance").<br>
        <span class='solution'>Pricing Strategy:</span><br>
        - ✅ <b>Segment Alpha</b>: 3-tiered plans (Starter, Professional, Growth) with free trial.<br>
        - ✅ <b>Segment Bravo</b>: Enterprise package with volume discounts, premium support, AUM-based pricing.<br>
        <span class='ceo-takeaway'>CEO Takeaway:</span> A dual GTM strategy drives rapid market penetration and long-term growth.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<b>Data-Driven Excellence</b>: All insights and visualizations are powered by  hiring data, client reviews, and strategic analysis done by wavess.io team, ensuring a robust foundation for multiply.ai's market strategy.
""", unsafe_allow_html=True)