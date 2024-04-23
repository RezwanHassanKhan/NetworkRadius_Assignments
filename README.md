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

Reach out to Rezwan at md.rezwanhassankhan@gmail.com for any further questions. :)
