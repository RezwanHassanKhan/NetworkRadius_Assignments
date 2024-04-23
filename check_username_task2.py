import radiusd

def authorize(p):
    try:
        # Retrieve User-Name attribute from the request
        username = [v[1] for v in p if v[0] == 'User-Name'][0]
    except IndexError:
        # User-Name not found
        radiusd.radlog(radiusd.L_ERR, 'User-Name not found in the request')
        return radiusd.RLM_MODULE_REJECT

    # Check if User-Name matches the specified pattern
   # if username == "bob":
    if username == "0\t\nbob":
        radiusd.radlog(radiusd.L_INFO, 'User-Name matches the expected value')
        return radiusd.RLM_MODULE_OK
    else:
        radiusd.radlog(radiusd.L_INFO, 'User-Name does not match the expected value')
        return radiusd.RLM_MODULE_REJECT
