"""Custom Report Builder Page."""

import streamlit as st
from datetime import datetime
from .base import BasePage
from src.report_builder import (
    create_report_template,
    build_custom_report,
    format_report_as_markdown,
    format_report_as_html
)


class CustomReportPage(BasePage):
    """Create customized reports with selected charts and data."""
    
    def __init__(self):
        super().__init__(
            "Custom Report Builder",
            "Create customized PDF-ready reports with selected charts and data."
        )
    
    def _render_content(self, df, config):
        """Render custom report builder."""
        countries = config.get('countries', [])
        metric_col = config.get('metric_col', 'GDP_Growth')
        
        if not countries:
            st.warning("Please select at least one country.")
            return
        
        self._render_report_config(df, countries, metric_col)
    
    def _render_report_config(self, df, countries, metric_col):
        """Render report configuration section."""
        st.subheader("Report Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Report Template",
                ["standard", "executive", "detailed"]
            )
        
        with col2:
            report_format = st.selectbox(
                "Output Format",
                ["markdown", "html"]
            )
        
        template = create_report_template(report_type)
        
        st.info(f"**{template['title']}**: {template['description']}")
        
        st.subheader("Select Sections")
        
        for i, section in enumerate(template['sections']):
            template['sections'][i]['include'] = st.checkbox(
                section['title'],
                value=section['include'],
                key=f"section_{section['id']}"
            )
        
        if st.button("Generate Report"):
            with st.spinner("Building custom report..."):
                report = build_custom_report(
                    df,
                    countries,
                    template,
                    metric_col=metric_col
                )
            
            if report:
                st.success("Report generated successfully!")
                
                if report_format == 'markdown':
                    content = format_report_as_markdown(report)
                    mime_type = "text/markdown"
                    file_ext = "md"
                else:
                    content = format_report_as_html(report)
                    mime_type = "text/html"
                    file_ext = "html"
                
                st.download_button(
                    label=f"Download Report ({file_ext.upper()})",
                    data=content,
                    file_name=f"gdp_report_{datetime.now().strftime('%Y%m%d')}.{file_ext}",
                    mime=mime_type
                )
                
                with st.expander("Preview Report"):
                    if report_format == 'markdown':
                        st.markdown(content)
                    else:
                        st.components.v1.html(content, height=800, scrolling=True)
            else:
                st.error("Unable to generate report")
