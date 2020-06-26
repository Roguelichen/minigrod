import ffai
import mybot
import argparse
import yaml

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', help='Configuration file for the competition', required=True)
args = parser.parse_args()

competition_config = yaml.safe_load(open(args.config, 'r'))
    
# Load configurations, rules, arena and teams
config = ffai.load_config(competition_config['config'])
ruleset = ffai.load_rule_set(config.ruleset)
arena = ffai.load_arena(config.arena)
team_a = ffai.load_team_by_filename(competition_config['competitor-a']['team'], ruleset)
team_b = ffai.load_team_by_filename(competition_config['competitor-b']['team'], ruleset)

# Make proxy agents
client_a = ffai.PythonSocketClient(competition_config['competitor-a']['name'], 
                                   host=competition_config['competitor-a']['hostname'], 
                                   port=competition_config['competitor-a']['port'], 
                                   token=competition_config['competitor-a']['token'])
client_b = ffai.PythonSocketClient(competition_config['competitor-b']['name'], 
                                   host=competition_config['competitor-b']['hostname'], 
                                   port=competition_config['competitor-b']['port'], 
                                   token=competition_config['competitor-b']['token'])

# Run competition
competition = ffai.Competition(client_a, client_b, team_a, team_b, config=config, ruleset=ruleset, arena=arena, n=competition_config['num_games'])
competition.run()
competition.results.print()
