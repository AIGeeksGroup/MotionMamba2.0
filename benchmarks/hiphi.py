"""Read the frozen hiphi control tasks without learned evaluator assets."""
def schedule(config,task,case):
    item=config['tasks'][task]
    phases=item['phases']
    return [case['root_states'][i] for i in phases],[case['captions'][i] for i in phases]
