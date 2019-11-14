import click

click.echo('Continue? [yn] ', nl=False)
c = click.getchar()
click.echo()
print(c)
if c == 'y':
    click.echo('We will go on')
elif c == 'n':
    click.echo('Abort!')
elif c =='àH':
    click.echo('UP!')
elif c =='àP':
    click.echo('DOWN!')
elif c =='àK':
    click.echo('LEFT!')
elif c =='àM':
    click.echo('RIGHT!')
else:
    click.echo('Invalid input :(')
