import click
@click.command()
@click.argument('data')
def pycaML(data, regression, classification, stacking, tuning):
    click.echo('Hello %s!' % data)


