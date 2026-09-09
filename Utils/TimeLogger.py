import datetime

logmsg = ''
timemark = dict()
saveDefault = False
_logfile = None

def set_logfile(path):
	"""Stream every logged line to `path`, starting with whatever is buffered.

	`logmsg` accumulated the whole run in memory and was never written anywhere.
	Appending per call instead of dumping at exit means a run that is killed --
	which is how long runs usually end -- still leaves its log behind.
	"""
	global _logfile
	_logfile = open(path, 'a', buffering=1)
	if logmsg:
		_logfile.write(logmsg)

def log(msg, save=None, oneline=False):
	global logmsg
	global saveDefault
	time = datetime.datetime.now()
	tem = '%s: %s' % (time, msg)
	if save != None:
		if save:
			logmsg += tem + '\n'
	elif saveDefault:
		logmsg += tem + '\n'
	if _logfile is not None:
		_logfile.write(tem + '\n')
	if oneline:
		print(tem, end='\r')
	else:
		print(tem)

def marktime(marker):
	global timemark
	timemark[marker] = datetime.datetime.now()


if __name__ == '__main__':
	log('')
