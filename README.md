# Python Script to Check User-Name in an Incoming Request

If the value matches "\0\t\nbob", the module should return the "ok" rcode. If
the value does not match or the User-Name attribute is not present, the module should return the
"reject" rcode.

How to install freeRadius in Ubuntu 22.04.04 LTS:
----------
### Update the system and install all dependecies

```console
1. sudo apt update
2. sudo apt install -y build-essential git libtalloc-dev\
   libssl-dev libpcre3-dev libcap-dev libncurses5-dev\
   libperl-dev libmysqlclient-dev libpq-dev libiodbc2-dev \
   libldap2-dev libkrb5-dev libcurl4-openssl-dev libhiredis-dev
```

### Clone the Repository
I used FreeRADIUS version 3.0.27 due to issues with version 4's obsolete libraries. Will migrate to v4 soon.
```console
1. git clone https://github.com/FreeRADIUS/freeradius-server.git
2. cd freeradius-server
```

### Configure the Build 
```console
1. ./configure --enable-developer
2. make
3. sudo make install
```
Important :
Sometime server.pem is not configured in ../raddb/certs/. So follo this step:
```console
1. cd \usr\local\etc\raddb\certs
2. make
3. sudo make install
```
### Run the Server in Debugging Mood
```console
radiusd -X
```
### Edit the Users Database  :
```console
1. cd /usr/local/etc/raddb
2. sudo vim users 
3. Uncomment this line : 
bob     Cleartext-Password := "hello"
        Reply-Message := "Hello, %{User-Name}"
```
### Do a quick test in radtest and radclient :
```console
1. radtest bob hello 127.0.0.1 0 testing123
2.  echo "User-Name = bob, User-Password = hello" | radclient localhost auth testing123
```
CONGRATULATIONS!!! SERVER IS UP AND ACCEPTING AND SENDING  REQUEST
### Building the python script "check_username.py" 

```console
1. cd /usr/local/etc/raddb
2. mkdir scripts
3. cd scripts
4. sudo vim check_username.py
```
In the check_username.py wrrite this script :
```python
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

```

### Configuring FreeRADIUS to Use the Python Script
If the Python module is not enabled in your FreeRADIUS setup, create a symbolic link from mods-available to mods-enabled:

```console
1. cd usr/local/etc/raddb/mods-enabled/
2. ln usr/local/etc/raddb/mods-available/python
```
### Configuring Python Module
```console
1. sudo vim usr/local/etc/raddb/mods-enabled/python
```
Write this statements in the python{} block:
```code
python {
       python_path = "/usr/local/etc/raddb/scripts"
       module = check_username
       mod_authorize = ${.module}
       func_authorize = authorize
}
```
### Authorize python module 
```console
1. sudo vim /usr/local/etc/raddb/sites-available/default
```
Write this statements in the authorize{} block:
```code
authorize {
       python
}
```
### Stop the server and Run Again
```console
1. sudo lsof -i :18120 to find the <PID>
2. sudo kill <PID>
3. radiusd -X
```
Please note that if you have error , stating python module is 
not found means that python is not installed . So
first check if rlm_python.so/rlm_python3.so is there in the directory:
```console
1. cd /usr/local/lib and do ls
```
if not found do this step to install python properly:
```console
1 ./configure --with-python2/3 
2. make
3. sudo make install
4. radiusd -X
```
## Testing with radclient
```console
1. echo "User-Name = \000\011\012bob, User-Password = hello" | radclient localhost auth testing123
```
## Result

### Test for Username = John Doe with password = hello and secret : testing123.\
Before test, change the policy for white_space from this location : sudo vim /usr/local/etc/raddb/policy.d/filter.\

radtest input : 
```console
1. radtest -x 'John Doe' hello 127.0.0.1 0 testing123.
`
``radtest output : 
```console
Sent Access-Request Id 106 from 0.0.0.0:56605 to 127.0.0.1:1812 length 78
	User-Name = "John Doe"
	User-Password = "hello"
	NAS-IP-Address = 127.0.1.1
	NAS-Port = 0
	Message-Authenticator = 0x00
	Cleartext-Password = "hello"
Received Access-Accept Id 106 from 127.0.0.1:1812 to 127.0.0.1:56605 length 37
	Reply-Message = "Hello, John Doe"
```
radiusd -X output :
```console
User-Name matches the expected value
(1)     [python] = ok
(1)     [chap] = noop
(1)     [mschap] = noop
(1)     [digest] = noop
(1) suffix: Checking for suffix after "@"
(1) suffix: No '@' in User-Name = "John Doe", looking up realm NULL
(1) suffix: No such realm "NULL"
(1)     [suffix] = noop
(1) eap: No EAP-Message, not doing EAP
(1)     [eap] = noop
(1) files: users: Matched entry John Doe at line 103
(1) files: EXPAND Hello, %{User-Name}
(1) files:    --> Hello, John Doe
(1)     [files] = ok
(1)     [expiration] = noop
(1)     [logintime] = noop
(1)     [pap] = updated
(1)   } # authorize = updated
(1) Found Auth-Type = PAP
(1) # Executing group from file /usr/local/etc/raddb/sites-enabled/default
(1)   Auth-Type PAP {
(1) pap: Login attempt with password
(1) pap: Comparing with "known good" Cleartext-Password
(1) pap: User authenticated successfully
(1)     [pap] = ok
(1)   } # Auth-Type PAP = ok
(1) # Executing section post-auth from file /usr/local/etc/raddb/sites-enabled/default
(1)   post-auth {
(1)     if (session-state:User-Name && reply:User-Name && request:User-Name && (reply:User-Name == request:User-Name)) {
(1)     if (session-state:User-Name && reply:User-Name && request:User-Name && (reply:User-Name == request:User-Name))  -> FALSE
(1)     update {
(1)       No attributes updated for RHS &session-state:
(1)     } # update = noop
(1)     [exec] = noop
(1)     policy remove_reply_message_if_eap {
(1)       if (&reply:EAP-Message && &reply:Reply-Message) {
(1)       if (&reply:EAP-Message && &reply:Reply-Message)  -> FALSE
(1)       else {
(1)         [noop] = noop
(1)       } # else = noop
(1)     } # policy remove_reply_message_if_eap = noop
(1)     if (EAP-Key-Name && &reply:EAP-Session-Id) {
(1)     if (EAP-Key-Name && &reply:EAP-Session-Id)  -> FALSE
(1)   } # post-auth = noop
(1) Sent Access-Accept Id 23 from 127.0.0.1:1812 to 127.0.0.1:36111 length 37
(1)   Reply-Message = "Hello, John Doe"
(1) Finished request

````
Reach out to Rezwan at md.rezwanhassankhan@gmail.com for any further questions. :)
