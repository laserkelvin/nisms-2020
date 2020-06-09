
from ruamel import yaml
from jinja2 import Template


# load in the schedule
with open("schedule.yml") as read_file:
    schedule = yaml.safe_load(read_file)
    
with open("schedule_template.html") as read_file:
    template = Template(read_file.read())

with open("schedule.html", "w+") as write_file:
    write_file.write(template.render(sessions=schedule))