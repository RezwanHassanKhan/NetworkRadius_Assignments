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

### 1 .Test for Username = \0\t\nbob or \x00\t\nbob with password = hellobob and secret : testing123.\
Sending user name as octal sequence : 
radtest input : 
```console
echo  'User-Name = \x00\011\012bob, User-Password = hellobob' | radclient -x localhost auth testing123
```
We are sunding \000 as \x00 or else radclient return the usename as "". 
radtest output : 
```console
Sent Access-Request Id 118 from 0.0.0.0:57029 to 127.0.0.1:1812 length 49
	User-Name = "\\x00\t\nbob"
	User-Password = "hellobob"
	Cleartext-Password = "hellobob"
Received Access-Reject Id 118 from 127.0.0.1:1812 to 127.0.0.1:57029 length 20
(0) -: Expected Access-Accept got Access-Reject"
```
radiusd -X output :\
<span style="color: red;"> When a username starts with a null character, `radclient` sends the escape sequence with an extra backslash. However, it correctly handles octal sequences that do not start with a null character, as demonstrated in Test 2.</span>

```console
('escape sequence username returned from radclien :', '\\x00\t\nbob')
('escape sequence username :', '\x00\t\nbob')
User-Name does not match the expected value
(0)     [python] = reject
(0)   } # authorize = reject
(0) Using Post-Auth-Type Reject
(0) # Executing group from file /usr/local/etc/raddb/sites-enabled/default
(0)   Post-Auth-Type REJECT {
(0) attr_filter.access_reject: EXPAND %{User-Name}
(0) attr_filter.access_reject:    --> \\x00\t\nbob
(0) attr_filter.access_reject: Matched entry DEFAULT at line 11
(0)     [attr_filter.access_reject] = updated
(0)     [eap] = noop
(0)     policy remove_reply_message_if_eap {
(0)       if (&reply:EAP-Message && &reply:Reply-Message) {
(0)       if (&reply:EAP-Message && &reply:Reply-Message)  -> FALSE
(0)       else {
(0)         [noop] = noop
(0)       } # else = noop
(0)     } # policy remove_reply_message_if_eap = noop
(0)   } # Post-Auth-Type REJECT = updated
(0) Delaying response for 1.000000 seconds
Waking up in 0.3 seconds.
Waking up in 0.6 seconds.
(0) Sending delayed response
(0) Sent Access-Reject Id 196 from 127.0.0.1:1812 to 127.0.0.1:42626 length 20
Waking up in 3.9 seconds.
(0) Cleaning up request packet ID 196 with timestamp +2 due to cleanup_delay was reached

````
### 2 .Test for Username = \t\t\nbob or  with password = hellobob and secret : testing123.\
Sending user name as octal sequence : 
radtest input : 
```console
echo  'User-Name = \011\011\012bob, User-Password = hellobob' | radclient -x localhost auth testing123
``` 
radtest output : \
Although the username matched, the access request was rejected. This user needs to be added to the users file with the appropriate policy settings for authorization.
```console
Sent Access-Request Id 76 from 0.0.0.0:60096 to 127.0.0.1:1812 length 46
	User-Name = "\t\t\nbob"
	User-Password = "hellobob"
	Cleartext-Password = "hellobob"
Received Access-Reject Id 76 from 127.0.0.1:1812 to 127.0.0.1:60096 length 20
(0) -: Expected Access-Accept got Access-Reject
```
radiusd -X output :\
Input User Name matches the expected valued but send an error 'ERROR: No Auth-Type found: rejecting the user via Post-Auth-Type = Reject', 
This is because inputted user  name was not found in  /user/local/etc/raddb/users and need to be inserted which is shown in Task 3 and 4. 
```console
('escape sequence username returned from radclient :', '\t\t\nbob')
('escape sequence username :', '\t\t\nbob')
User-Name matches the expected value
(0)     [python] = ok
(0)     [chap] = noop
(0)     [mschap] = noop
(0)     [digest] = noop
(0) suffix: Checking for suffix after "@"
(0) suffix: No '@' in User-Name = "		 bob", looking up realm NULL
(0) suffix: No such realm "NULL"
(0)     [suffix] = noop
(0) eap: No EAP-Message, not doing EAP
(0)     [eap] = noop
(0)     [files] = noop
(0)     [expiration] = noop
(0)     [logintime] = noop
(0) pap: WARNING: No "known good" password found for the user.  Not setting Auth-Type
(0) pap: WARNING: Authentication will fail unless a "known good" password is available
(0)     [pap] = noop
(0)   } # authorize = ok
(0) ERROR: No Auth-Type found: rejecting the user via Post-Auth-Type = Reject
(0) Failed to authenticate the user
(0) Using Post-Auth-Type Reject
(0) # Executing group from file /usr/local/etc/raddb/sites-enabled/default
(0)   Post-Auth-Type REJECT {
(0) attr_filter.access_reject: EXPAND %{User-Name}
(0) attr_filter.access_reject:    --> \t\t\nbob
(0) attr_filter.access_reject: Matched entry DEFAULT at line 11
(0)     [attr_filter.access_reject] = updated
(0)     [eap] = noop
(0)     policy remove_reply_message_if_eap {
(0)       if (&reply:EAP-Message && &reply:Reply-Message) {
(0)       if (&reply:EAP-Message && &reply:Reply-Message)  -> FALSE
(0)       else {
(0)         [noop] = noop
(0)       } # else = noop
(0)     } # policy remove_reply_message_if_eap = noop
(0)   } # Post-Auth-Type REJECT = updated
(0) Delaying response for 1.000000 seconds
Waking up in 0.3 seconds.
Waking up in 0.6 seconds.
(0) Sending delayed response
(0) Sent Access-Reject Id 76 from 127.0.0.1:1812 to 127.0.0.1:60096 length 20
Waking up in 3.9 seconds.
(0) Cleaning up request packet ID 76 with timestamp +12 due to cleanup_delay was reached
````


### 2 .Test for Username = bob with password = hellobob and secret : testing123.\

Before test, change the policy for white_space from this location : sudo vim /usr/local/etc/raddb/policy.d/filter.\

radtest input : 
```console
1. radtest -x 'bob' hellobob 127.0.0.1 0 testing123
```
radtest output : 
```console
Sent Access-Request Id 11 from 0.0.0.0:52876 to 127.0.0.1:1812 length 73
	User-Name = "bob"
	User-Password = "hellobob"
	NAS-IP-Address = 127.0.1.1
	NAS-Port = 0
	Message-Authenticator = 0x00
	Cleartext-Password = "hellobob"
Received Access-Accept Id 11 from 127.0.0.1:1812 to 127.0.0.1:52876 length 32
	Reply-Message = "Hello, bob"
```
radiusd -X output :
```console
User-Name matches the expected value
(0)     [python] = ok
(0)     [chap] = noop
(0)     [mschap] = noop
(0)     [digest] = noop
(0) suffix: Checking for suffix after "@"
(0) suffix: No '@' in User-Name = "bob", looking up realm NULL
(0) suffix: No such realm "NULL"
(0)     [suffix] = noop
(0) eap: No EAP-Message, not doing EAP
(0)     [eap] = noop
(0) files: users: Matched entry bob at line 92
(0) files: EXPAND Hello, %{User-Name}
(0) files:    --> Hello, bob
(0)     [files] = ok
(0)     [expiration] = noop
(0)     [logintime] = noop
(0)     [pap] = updated
(0)   } # authorize = updated
(0) Found Auth-Type = PAP
(0) # Executing group from file /usr/local/etc/raddb/sites-enabled/default
(0)   Auth-Type PAP {
(0) pap: Login attempt with password
(0) pap: Comparing with "known good" Cleartext-Password
(0) pap: User authenticated successfully
(0)     [pap] = ok
(0)   } # Auth-Type PAP = ok
(0) # Executing section post-auth from file /usr/local/etc/raddb/sites-enabled/default
(0)   post-auth {
(0)     if (session-state:User-Name && reply:User-Name && request:User-Name && (reply:User-Name == request:User-Name)) {
(0)     if (session-state:User-Name && reply:User-Name && request:User-Name && (reply:User-Name == request:User-Name))  -> FALSE
(0)     update {
(0)       No attributes updated for RHS &session-state:
(0)     } # update = noop
(0)     [exec] = noop
(0)     policy remove_reply_message_if_eap {
(0)       if (&reply:EAP-Message && &reply:Reply-Message) {
(0)       if (&reply:EAP-Message && &reply:Reply-Message)  -> FALSE
(0)       else {
(0)         [noop] = noop
(0)       } # else = noop
(0)     } # policy remove_reply_message_if_eap = noop
(0)     if (EAP-Key-Name && &reply:EAP-Session-Id) {
(0)     if (EAP-Key-Name && &reply:EAP-Session-Id)  -> FALSE
(0)   } # post-auth = noop
(0) Sent Access-Accept Id 11 from 127.0.0.1:1812 to 127.0.0.1:52876 length 32
(0)   Reply-Message = "Hello, bob"
(0) Finished request
```
### 3.Test for Username = John Doe with password = hello and secret : testing123.\

Before test, change the policy for white_space from this location : sudo vim /usr/local/etc/raddb/policy.d/filter.\

radtest input : 
```console
1. radtest -x 'John Doe' hello 127.0.0.1 0 testing123
```
radtest output : 
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


