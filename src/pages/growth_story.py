"""Growth Story Generator Page."""

import streamlit as st
import pandas as pd
from .base import BasePage
from src.story_generator import (
    generate_growth_story,
    generate_comparative_story,
    format_story_as_text
)


class GrowthStoryPage(BasePage):
    """Auto-generate narrative summaries of country economic performance."""
    
    def __init__(self):
        super().__init__(
            "Growth Story Generator",
            "Auto-generate narrative summaries of country economic performance."
        )
    
    def _render_content(self, df, config):
        """Render growth story generation."""
        countries = config.get('countries', [])
        metric_col = config.get('metric_col', 'GDP_Growth')
        
        if not countries:
            st.warning("Please select at least one country.")
            return
        
        story_type = st.radio("Story Type", ["Individual Country", "Comparative Analysis"])
        
        if story_type == "Individual Country":
            self._render_individual_story(df, countries, metric_col)
        else:
            self._render_comparative_story(df, countries, metric_col)
    
    def _render_individual_story(self, df, countries, metric_col):
        """Render individual country story."""
        selected_country = st.selectbox("Select country", countries)
        
        if st.button("Generate Story"):
            with st.spinner(f"Generating growth story for {selected_country}..."):
                story = generate_growth_story(df, selected_country, metric_col=metric_col)
            
            if story:
                st.subheader(story['title'])
                
                for section in story['sections']:
                    with st.expander(section['title'], expanded=True):
                        st.write(section['content'])
                
                st.download_button(
                    label="Download Story (Markdown)",
                    data=format_story_as_text(story),
                    file_name=f"growth_story_{selected_country}.md",
                    mime="text/markdown"
                )
            else:
                st.error("Unable to generate story")
    
    def _render_comparative_story(self, df, countries, metric_col):
        """Render comparative story."""
        if len(countries) < 2:
            st.warning("Please select at least 2 countries for comparative story.")
            return
        
        if st.button("Generate Comparative Story"):
            with st.spinner("Generating comparative story..."):
                story = generate_comparative_story(df, countries, metric_col=metric_col)
            
            if story:
                st.subheader(story['title'])
                
                for section in story['sections']:
                    with st.expander(section['title'], expanded=True):
                        st.write(section['content'])
                
                st.download_button(
                    label="Download Story (Markdown)",
                    data=format_story_as_text(story),
                    file_name="comparative_growth_story.md",
                    mime="text/markdown"
                )
            else:
                st.error("Unable to generate comparative story")
