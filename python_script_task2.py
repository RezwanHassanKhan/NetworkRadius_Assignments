import radiusd

def authorize(p):
    try:
        # Retrieve User-Name attribute from the request
        username = [v[1] for v in p if v[0] == 'User-Name'][0]
    except IndexError:
        # User-Name not found
        radiusd.radlog(radiusd.L_ERR, 'User-Name not found in the request')
        return radiusd.RLM_MODULE_REJECT
  
    #Check if User-Name matches the specified pattern
    #Some test usernames being tested are: Bob, John Doe, \t\t\nbob.
    
    #This is excape  sequence equilvalent to \x00\011\012bob in octal sequence form
    expected_username = "\x00\t\nbob"

    print('escape sequence username returned from radclient :', username)
    print('escape sequence username :', expected_username)
    
    if username == expected_username :
        radiusd.radlog(radiusd.L_INFO, 'User-Name matches the expected value')
        return radiusd.RLM_MODULE_OK
    else:
        radiusd.radlog(radiusd.L_INFO, 'User-Name does not match the expected value')
        return radiusd.RLM_MODULE_REJECT
