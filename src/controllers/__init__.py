from ..controllers.upload import index

url_routes = [
    {
        'rule': '/',
        'endpoint': 'index',
        'view_func': index,
        'methods': ['GET', 'POST']
    }
]