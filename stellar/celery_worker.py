from uuid import uuid4
import click
from celery_app import celery_app
from celery_tasks import get_clients

@click.command()
@click.option('--concurrency', type=int, default=16)
@click.option('--loglevel', type=str, default='info')
def server(concurrency: int, loglevel: str):

    uuid = str(uuid4().hex)
    
    celery_app.worker_main(['worker', '-n', f'server_{uuid}@%h', '-Q', 'server', '--concurrency', str(concurrency), '--loglevel', loglevel])

@click.command()
@click.argument('client_id', type=str)
@click.option('--concurrency', type=int, default=2)
@click.option('--loglevel', type=str, default='info')
def client(client_id: str, concurrency: int, loglevel: str):
    from task_state_manager import get_task_state_manager
    
    state = get_task_state_manager()
    if client_id in get_clients().values():
        print(f"Client {client_id} already exists")
        return

    uuid = str(uuid4().hex)
    
    celery_app.worker_main(['worker', '-n', f'client_{uuid}@%h', '-Q', f'client_{client_id}', '--concurrency', str(concurrency), '--loglevel', loglevel])

@click.command()
def test():
    from task_state_manager import get_task_state_manager
    state = get_task_state_manager()
    print(state.get_devices())
    
    # print(state.get_task('f69fdb4a-e946-4cce-86cc-265a31b12927'))
    
    inspector = celery_app.control.inspect()
    active_workers = inspector.stats()

    if active_workers:
        print("Active Celery Workers:")
        for worker_name, stats in active_workers.items():
            worker_inspector = celery_app.control.inspect([worker_name])
            active_queues = worker_inspector.active_queues()[worker_name]
            print(f"- {worker_name}")
            print(f"  - {stats}")
            print(f"  - {[queue['routing_key'] for queue in active_queues]}")
    else:
        print("No active Celery workers found.")

@click.group()
def cli():
    pass

cli.add_command(client)
cli.add_command(server)
cli.add_command(test)

if __name__ == '__main__':
    cli()
