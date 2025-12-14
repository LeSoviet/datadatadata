import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from datetime import datetime


def create_report_template(template_type='standard'):
    templates = {
        'standard': {
            'title': 'GDP Growth Analysis Report',
            'description': 'Standard report with essential metrics and visualizations',
            'sections': [
                {'id': 'overview', 'title': 'Executive Summary', 'include': True},
                {'id': 'time_series', 'title': 'Time Series Analysis', 'include': True},
                {'id': 'statistics', 'title': 'Statistical Summary', 'include': True},
                {'id': 'comparison', 'title': 'Country Comparison', 'include': True}
            ]
        },
        'executive': {
            'title': 'Executive Briefing - GDP Growth',
            'description': 'Concise summary for executive decision-makers',
            'sections': [
                {'id': 'overview', 'title': 'Key Findings', 'include': True},
                {'id': 'highlights', 'title': 'Highlights', 'include': True},
                {'id': 'comparison', 'title': 'Quick Comparison', 'include': True}
            ]
        },
        'detailed': {
            'title': 'Comprehensive GDP Growth Report',
            'description': 'In-depth analysis with all available metrics',
            'sections': [
                {'id': 'overview', 'title': 'Executive Summary', 'include': True},
                {'id': 'time_series', 'title': 'Historical Analysis', 'include': True},
                {'id': 'statistics', 'title': 'Statistical Analysis', 'include': True},
                {'id': 'highlights', 'title': 'Key Highlights', 'include': True},
                {'id': 'volatility', 'title': 'Volatility & Risk Assessment', 'include': True},
                {'id': 'comparison', 'title': 'Comparative Analysis', 'include': True},
                {'id': 'forecast', 'title': 'Forward-Looking Analysis', 'include': True},
                {'id': 'regime', 'title': 'Growth Regime Classification', 'include': True}
            ]
        }
    }
    
    return templates.get(template_type, templates['standard'])


def generate_overview_section(df, countries, metric_col='GDP_Growth'):
    filtered = df[df['Entity'].isin(countries)]
    
    overview = {
        'countries': countries,
        'total_countries': len(countries),
        'year_range': f"{int(filtered['Year'].min())}-{int(filtered['Year'].max())}",
        'avg_growth': filtered[metric_col].mean(),
        'total_observations': len(filtered)
    }
    
    overview_text = f"This report analyzes GDP growth for {len(countries)} countries "
    overview_text += f"covering the period {overview['year_range']}. "
    overview_text += f"Average growth across all selected countries is {overview['avg_growth']:.2f}%."
    
    return {
        'type': 'overview',
        'data': overview,
        'text': overview_text
    }


def generate_statistics_section(df, countries, metric_col='GDP_Growth'):
    stats_list = []
    
    for country in countries:
        country_data = df[df['Entity'] == country][metric_col].dropna()
        
        if len(country_data) == 0:
            continue
        
        stats = {
            'Country': country,
            'Mean': country_data.mean(),
            'Median': country_data.median(),
            'Std Dev': country_data.std(),
            'Min': country_data.min(),
            'Max': country_data.max(),
            'Range': country_data.max() - country_data.min()
        }
        
        stats_list.append(stats)
    
    stats_df = pd.DataFrame(stats_list)
    
    return {
        'type': 'statistics',
        'data': stats_df
    }


def generate_time_series_chart(df, countries, metric_col='GDP_Growth'):
    filtered = df[df['Entity'].isin(countries)].copy()
    
    fig = px.line(
        filtered,
        x='Year',
        y=metric_col,
        color='Entity',
        title='GDP Growth Over Time'
    )
    
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='GDP Growth (%)',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig


def generate_comparison_chart(df, countries, metric_col='GDP_Growth'):
    stats_list = []
    
    for country in countries:
        country_data = df[df['Entity'] == country][metric_col].dropna()
        
        if len(country_data) > 0:
            stats_list.append({
                'Country': country,
                'Average Growth': country_data.mean(),
                'Volatility': country_data.std()
            })
    
    stats_df = pd.DataFrame(stats_list)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=stats_df['Country'],
        y=stats_df['Average Growth'],
        name='Average Growth',
        marker_color='#374151'
    ))
    
    fig.update_layout(
        title='Average Growth by Country',
        xaxis_title='Country',
        yaxis_title='Average Growth (%)',
        showlegend=True
    )
    
    return fig


def build_custom_report(df, countries, template, metric_col='GDP_Growth', charts=None):
    report = {
        'title': template['title'],
        'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'countries': countries,
        'sections': []
    }
    
    for section_config in template['sections']:
        if not section_config['include']:
            continue
        
        section_id = section_config['id']
        section_title = section_config['title']
        
        section_data = None
        
        if section_id == 'overview':
            section_data = generate_overview_section(df, countries, metric_col)
        
        elif section_id == 'statistics':
            section_data = generate_statistics_section(df, countries, metric_col)
        
        elif section_id == 'highlights':
            section_data = generate_highlights_section(df, countries, metric_col)
        
        elif section_id == 'time_series':
            section_data = {
                'type': 'chart',
                'chart': generate_time_series_chart(df, countries, metric_col),
                'chart_type': 'time_series'
            }
        
        elif section_id == 'comparison':
            section_data = {
                'type': 'chart',
                'chart': generate_comparison_chart(df, countries, metric_col),
                'chart_type': 'comparison'
            }
        
        elif section_id == 'volatility':
            section_data = generate_volatility_section(df, countries, metric_col)
        
        elif section_id == 'forecast':
            section_data = generate_forecast_section(df, countries, metric_col)
        
        elif section_id == 'regime':
            section_data = generate_regime_section(df, countries, metric_col)
        
        if section_data:
            report['sections'].append({
                'id': section_id,
                'title': section_title,
                'data': section_data
            })
    
    return report


def generate_highlights_section(df, countries, metric_col='GDP_Growth'):
    highlights = []
    
    for country in countries:
        country_data = df[df['Entity'] == country][metric_col].dropna()
        if len(country_data) > 0:
            max_growth = country_data.max()
            highlights.append(f"**{country}**: Peak growth of {max_growth:.2f}%")
    
    highlights_text = "\n".join(highlights)
    
    return {
        'type': 'text',
        'text': highlights_text
    }


def generate_volatility_section(df, countries, metric_col='GDP_Growth'):
    volatility_data = []
    
    for country in countries:
        country_data = df[df['Entity'] == country][metric_col].dropna()
        if len(country_data) > 0:
            volatility_data.append({
                'Country': country,
                'Volatility (Std Dev)': country_data.std(),
                'Coefficient of Variation': (country_data.std() / abs(country_data.mean())) if country_data.mean() != 0 else 0
            })
    
    volatility_df = pd.DataFrame(volatility_data)
    
    return {
        'type': 'statistics',
        'data': volatility_df
    }


def generate_forecast_section(df, countries, metric_col='GDP_Growth'):
    forecast_text = "Forecast analysis placeholder. Advanced forecasting models can be applied to generate future projections."
    
    return {
        'type': 'text',
        'text': forecast_text
    }


def generate_regime_section(df, countries, metric_col='GDP_Growth'):
    regime_text = "Growth regime classification placeholder. Countries can be classified into different growth regimes based on their historical patterns."
    
    return {
        'type': 'text',
        'text': regime_text
    }


def format_report_as_markdown(report):
    md = f"# {report['title']}\n\n"
    md += f"**Generated**: {report['generated_date']}\n\n"
    md += f"**Countries**: {', '.join(report['countries'])}\n\n"
    md += "---\n\n"
    
    for section in report['sections']:
        md += f"## {section['title']}\n\n"
        
        data_type = section['data']['type']
        
        if data_type == 'overview':
            md += section['data']['text'] + "\n\n"
        
        elif data_type == 'text':
            md += section['data']['text'] + "\n\n"
        
        elif data_type == 'statistics':
            df = section['data']['data']
            md += df.to_markdown(index=False) + "\n\n"
        
        elif data_type == 'chart':
            chart_type = section['data'].get('chart_type', 'unknown')
            md += f"*[{chart_type.replace('_', ' ').title()} Chart]*\n\n"
            
            fig = section['data']['chart']
            try:
                img_base64 = fig.to_image(format='png', engine='kaleido')
                md += f"![{section['title']}](data:image/png;base64,{img_base64})\n\n"
            except:
                md += f"*Chart visualization available in HTML format*\n\n"
    
    md += "---\n\n"
    md += "*Generated by GDP Growth Analysis Platform*\n"
    
    return md


def format_report_as_html(report):
    html = f"<html><head><title>{report['title']}</title>"
    html += "<style>"
    html += "body { font-family: Inter, Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; background: #fafafa; }"
    html += "h1 { color: #1f2937; border-bottom: 2px solid #374151; padding-bottom: 10px; }"
    html += "h2 { color: #374151; margin-top: 30px; border-bottom: 1px solid #d1d5db; padding-bottom: 5px; }"
    html += "table { width: 100%; border-collapse: collapse; margin: 20px 0; background: white; }"
    html += "th, td { border: 1px solid #d1d5db; padding: 10px; text-align: left; }"
    html += "th { background-color: #f3f4f6; font-weight: 600; }"
    html += ".meta { color: #6b7280; font-size: 0.9em; margin-bottom: 20px; }"
    html += ".chart-container { margin: 20px 0; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }"
    html += "p { line-height: 1.6; color: #374151; }"
    html += "</style></head><body>"
    
    html += f"<h1>{report['title']}</h1>"
    html += f"<div class='meta'><strong>Generated:</strong> {report['generated_date']}<br>"
    html += f"<strong>Countries:</strong> {', '.join(report['countries'])}</div>"
    html += "<hr>"
    
    for section in report['sections']:
        html += f"<h2>{section['title']}</h2>"
        
        data_type = section['data']['type']
        
        if data_type == 'overview':
            html += f"<p>{section['data']['text']}</p>"
        
        elif data_type == 'text':
            html += f"<p>{section['data']['text'].replace(chr(10), '<br>')}</p>"
        
        elif data_type == 'statistics':
            df = section['data']['data']
            html += df.to_html(index=False, border=0, classes='stats-table')
        
        elif data_type == 'chart':
            fig = section['data']['chart']
            try:
                chart_html = fig.to_html(include_plotlyjs='cdn', div_id=f"chart_{section['id']}")
                html += f"<div class='chart-container'>{chart_html}</div>"
            except Exception as e:
                html += f"<div class='chart-container'><p style='color: #ef4444;'>Chart rendering failed: {str(e)}</p></div>"
    
    html += "<hr><p style='text-align: center; color: #6b7280; font-size: 0.9em;'>"
    html += "Generated by GDP Growth Analysis Platform</p>"
    html += "</body></html>"
    
    return html


def export_charts_to_images(charts, format='png'):
    chart_images = {}
    
    for chart_name, fig in charts.items():
        img_bytes = fig.to_image(format=format)
        chart_images[chart_name] = img_bytes
    
    return chart_images


def create_report_bundle(report, charts=None, format='markdown'):
    bundle = {
        'report': report,
        'format': format
    }
    
    if format == 'markdown':
        bundle['content'] = format_report_as_markdown(report)
    elif format == 'html':
        bundle['content'] = format_report_as_html(report)
    
    if charts:
        bundle['charts'] = charts
    
    return bundle


def save_report_to_file(report, filepath, format='markdown'):
    if format == 'markdown':
        content = format_report_as_markdown(report)
    elif format == 'html':
        content = format_report_as_html(report)
    else:
        content = str(report)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filepath
