def confirm_with_user(message):
    print('Are you sure? (y/n)')
    response = input()
    return response.lower() == 'y'


def get_md5(string):
    """Get the md5 hash of a string"""
    import hashlib
    return hashlib.md5(string.encode()).hexdigest()

def get_hash(args):
    """Get a hash of the arguments"""
    return get_md5(str(args))