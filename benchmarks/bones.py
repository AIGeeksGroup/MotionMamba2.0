"""Read the frozen bones control tasks without learned evaluator assets."""
def schedule(config,task,case):
    item=config['tasks'][task]
    commands=[name for name,count in item['stages'] for _ in range(count)]
    return commands,[config['prompts'][name] for name in commands]
