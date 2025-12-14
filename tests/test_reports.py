from src.report_builder import create_report_template

templates = ['standard', 'executive', 'detailed']

for template_type in templates:
    template = create_report_template(template_type)
    print(f"\n{template_type.upper()}: {template['title']}")
    print(f"Description: {template['description']}")
    print(f"Sections: {len(template['sections'])}")
    for section in template['sections']:
        print(f"  - {section['title']} (ID: {section['id']}, Include: {section['include']})")
