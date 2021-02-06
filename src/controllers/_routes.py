url_routes = []

def add_endpoint(rule, endpoint, f, **options):
    url_routes.append({
        'rule': rule,
        'endpoint': endpoint,
        'view_func': f,
        'methods': options.get('methods', ['GET'])
    })
    print("Registered rule: {} for endpoint {}".format(rule, endpoint))

def endpoint(rule, endpoint, **options):
    def decorator(f):
        add_endpoint(rule, endpoint, f, **options)
        return f

    return decorator