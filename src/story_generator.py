import pandas as pd
import numpy as np


def generate_growth_story(df, country, metric_col='GDP_Growth'):
    country_data = df[df['Entity'] == country].copy()
    
    if len(country_data) == 0:
        return None
    
    country_data = country_data.sort_values('Year')
    growth = country_data[metric_col].dropna()
    
    if len(growth) == 0:
        return None
    
    story = {
        'country': country,
        'title': f"Economic Growth Story: {country}",
        'sections': []
    }
    
    overview = generate_overview(country_data, metric_col)
    story['sections'].append(overview)
    
    phases = identify_growth_phases(country_data, metric_col)
    if phases:
        story['sections'].append(phases)
    
    highlights = identify_highlights(country_data, metric_col)
    if highlights:
        story['sections'].append(highlights)
    
    challenges = identify_challenges(country_data, metric_col)
    if challenges:
        story['sections'].append(challenges)
    
    recent_performance = analyze_recent_performance(country_data, metric_col)
    if recent_performance:
        story['sections'].append(recent_performance)
    
    outlook = generate_outlook(country_data, metric_col)
    if outlook:
        story['sections'].append(outlook)
    
    return story


def generate_overview(country_data, metric_col):
    growth = country_data[metric_col].dropna()
    
    years_span = f"{int(country_data['Year'].min())}-{int(country_data['Year'].max())}"
    avg_growth = growth.mean()
    
    if avg_growth > 5:
        performance = "strong"
    elif avg_growth > 3:
        performance = "solid"
    elif avg_growth > 1:
        performance = "moderate"
    elif avg_growth > 0:
        performance = "slow"
    else:
        performance = "challenging"
    
    overview_text = f"From {years_span}, {country_data['Entity'].iloc[0]} demonstrated {performance} economic performance "
    overview_text += f"with an average GDP growth rate of {avg_growth:.2f}% per year. "
    
    volatility = growth.std()
    if volatility > 5:
        overview_text += "The economy experienced high volatility, indicating significant fluctuations in growth."
    elif volatility > 3:
        overview_text += "Growth showed moderate variability, reflecting typical economic cycles."
    else:
        overview_text += "The economy maintained relatively stable growth patterns."
    
    return {
        'type': 'overview',
        'title': 'Overview',
        'content': overview_text
    }


def identify_growth_phases(country_data, metric_col):
    growth = country_data[metric_col].dropna()
    
    if len(growth) < 5:
        return None
    
    phases = []
    
    window = 5
    for i in range(0, len(growth) - window + 1, window):
        phase_data = growth.iloc[i:i+window]
        phase_years = country_data['Year'].iloc[i:i+window]
        
        avg_phase_growth = phase_data.mean()
        
        phase_start = int(phase_years.iloc[0])
        phase_end = int(phase_years.iloc[-1])
        
        if avg_phase_growth > 5:
            description = f"rapid expansion (avg {avg_phase_growth:.1f}%)"
        elif avg_phase_growth > 3:
            description = f"robust growth (avg {avg_phase_growth:.1f}%)"
        elif avg_phase_growth > 1:
            description = f"moderate growth (avg {avg_phase_growth:.1f}%)"
        elif avg_phase_growth > 0:
            description = f"slow growth (avg {avg_phase_growth:.1f}%)"
        else:
            description = f"contraction (avg {avg_phase_growth:.1f}%)"
        
        phases.append(f"{phase_start}-{phase_end}: {description}")
    
    phases_text = "The economic trajectory can be divided into distinct phases: " + "; ".join(phases) + "."
    
    return {
        'type': 'phases',
        'title': 'Growth Phases',
        'content': phases_text
    }


def identify_highlights(country_data, metric_col):
    growth = country_data[metric_col].dropna()
    
    if len(growth) == 0:
        return None
    
    highlights = []
    
    max_growth = growth.max()
    max_year = int(country_data.loc[growth.idxmax(), 'Year'])
    highlights.append(f"Peak growth of {max_growth:.2f}% in {max_year}")
    
    top_5_years = growth.nlargest(5)
    if len(top_5_years) >= 3:
        consecutive = []
        years = sorted([int(country_data.loc[idx, 'Year']) for idx in top_5_years.index])
        
        for i in range(len(years) - 1):
            if years[i+1] - years[i] == 1:
                if len(consecutive) == 0:
                    consecutive.append(years[i])
                consecutive.append(years[i+1])
        
        if len(consecutive) >= 3:
            highlights.append(f"Sustained high performance during {consecutive[0]}-{consecutive[-1]}")
    
    positive_years = (growth > 0).sum()
    total_years = len(growth)
    positive_pct = (positive_years / total_years) * 100
    
    if positive_pct > 90:
        highlights.append(f"Exceptional consistency with positive growth in {positive_pct:.0f}% of years")
    elif positive_pct > 75:
        highlights.append(f"Strong consistency with positive growth in {positive_pct:.0f}% of years")
    
    highlights_text = "Key highlights: " + "; ".join(highlights) + "."
    
    return {
        'type': 'highlights',
        'title': 'Key Achievements',
        'content': highlights_text
    }


def identify_challenges(country_data, metric_col):
    growth = country_data[metric_col].dropna()
    
    if len(growth) == 0:
        return None
    
    challenges = []
    
    min_growth = growth.min()
    min_year = int(country_data.loc[growth.idxmin(), 'Year'])
    
    if min_growth < -3:
        challenges.append(f"severe contraction of {min_growth:.2f}% in {min_year}")
    elif min_growth < 0:
        challenges.append(f"economic downturn of {min_growth:.2f}% in {min_year}")
    
    negative_years = (growth < 0).sum()
    if negative_years > 0:
        challenges.append(f"{negative_years} years of negative growth")
    
    volatility = growth.std()
    if volatility > 5:
        challenges.append(f"high volatility (σ={volatility:.1f}%)")
    
    consecutive_negative = 0
    max_consecutive_negative = 0
    for g in growth:
        if g < 0:
            consecutive_negative += 1
            max_consecutive_negative = max(max_consecutive_negative, consecutive_negative)
        else:
            consecutive_negative = 0
    
    if max_consecutive_negative >= 2:
        challenges.append(f"prolonged recession ({max_consecutive_negative} consecutive years of contraction)")
    
    if not challenges:
        return None
    
    challenges_text = "Economic challenges faced: " + "; ".join(challenges) + "."
    
    return {
        'type': 'challenges',
        'title': 'Challenges',
        'content': challenges_text
    }


def analyze_recent_performance(country_data, metric_col, years=5):
    growth = country_data[metric_col].dropna()
    
    if len(growth) < years:
        return None
    
    recent_growth = growth.tail(years)
    historical_growth = growth.head(-years) if len(growth) > years else growth
    
    recent_avg = recent_growth.mean()
    historical_avg = historical_growth.mean()
    
    if recent_avg > historical_avg + 1:
        trend = "accelerating"
    elif recent_avg < historical_avg - 1:
        trend = "decelerating"
    else:
        trend = "stable"
    
    recent_text = f"Recent performance ({years} years): {trend} with average growth of {recent_avg:.2f}% "
    recent_text += f"compared to historical average of {historical_avg:.2f}%. "
    
    recent_volatility = recent_growth.std()
    if recent_volatility > 3:
        recent_text += "Recent volatility remains elevated."
    elif recent_volatility < 2:
        recent_text += "Recent stability has improved."
    else:
        recent_text += "Volatility remains moderate."
    
    return {
        'type': 'recent',
        'title': 'Recent Performance',
        'content': recent_text
    }


def generate_outlook(country_data, metric_col):
    growth = country_data[metric_col].dropna()
    
    if len(growth) < 3:
        return None
    
    recent_3y = growth.tail(3).mean()
    overall_avg = growth.mean()
    
    if recent_3y > overall_avg + 1:
        outlook = "positive momentum suggests continued expansion"
    elif recent_3y < overall_avg - 1:
        outlook = "recent slowdown indicates need for policy adjustments"
    else:
        outlook = "stable trajectory expected to continue"
    
    recent_trend = (growth.iloc[-1] - growth.iloc[-3]) / 3 if len(growth) >= 3 else 0
    
    if recent_trend > 1:
        outlook += " with upward trend"
    elif recent_trend < -1:
        outlook += " despite downward pressures"
    
    outlook_text = f"Looking forward, {outlook}."
    
    return {
        'type': 'outlook',
        'title': 'Outlook',
        'content': outlook_text
    }


def generate_comparative_story(df, countries, metric_col='GDP_Growth'):
    if len(countries) < 2:
        return None
    
    story = {
        'title': f"Comparative Analysis: {', '.join(countries)}",
        'sections': []
    }
    
    all_data = []
    for country in countries:
        country_data = df[df['Entity'] == country].copy()
        growth = country_data[metric_col].dropna()
        
        if len(growth) > 0:
            all_data.append({
                'country': country,
                'avg_growth': growth.mean(),
                'volatility': growth.std(),
                'max_growth': growth.max(),
                'min_growth': growth.min()
            })
    
    if len(all_data) == 0:
        return None
    
    all_data_sorted = sorted(all_data, key=lambda x: x['avg_growth'], reverse=True)
    
    leader = all_data_sorted[0]
    laggard = all_data_sorted[-1]
    
    comparison_text = f"Among the selected countries, {leader['country']} leads with an average growth of {leader['avg_growth']:.2f}%, "
    comparison_text += f"while {laggard['country']} shows the lowest average at {laggard['avg_growth']:.2f}%. "
    
    most_volatile = max(all_data, key=lambda x: x['volatility'])
    most_stable = min(all_data, key=lambda x: x['volatility'])
    
    comparison_text += f"{most_volatile['country']} experienced the highest volatility (σ={most_volatile['volatility']:.1f}%), "
    comparison_text += f"whereas {most_stable['country']} maintained the most stable path (σ={most_stable['volatility']:.1f}%)."
    
    story['sections'].append({
        'type': 'comparison',
        'title': 'Comparative Overview',
        'content': comparison_text
    })
    
    return story


def format_story_as_text(story):
    if not story:
        return "No story available."
    
    text = f"# {story['title']}\n\n"
    
    for section in story['sections']:
        text += f"## {section['title']}\n\n"
        text += f"{section['content']}\n\n"
    
    return text


def format_story_as_html(story):
    if not story:
        return "<p>No story available.</p>"
    
    html = f"<h1>{story['title']}</h1>"
    
    for section in story['sections']:
        html += f"<h2>{section['title']}</h2>"
        html += f"<p>{section['content']}</p>"
    
    return html
