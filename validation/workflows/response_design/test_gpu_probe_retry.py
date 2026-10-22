import subprocess,unittest
from gpu_probe_retry import retry_run
class RetryTests(unittest.TestCase):
 def test_timeout_and_nonzero_are_waiting_not_failure(self):
  count=[];sleeps=[]
  def run(command,**kw):
   count.append(1)
   if len(count)<=3:raise subprocess.TimeoutExpired(command,45)
   if len(count)==4:return subprocess.CompletedProcess(command,1,'','temporary failure')
   return subprocess.CompletedProcess(command,0,'0, 0, 0\n','')
  r=retry_run(['srun','nvidia-smi'],runner=run,sleeper=sleeps.append)
  self.assertEqual(len(sleeps),4);self.assertEqual(sleeps,[60]*4);self.assertEqual(r.returncode,0)
 def test_physical_timeout_is_not_retried(self):
  def run(command,**kw):raise subprocess.TimeoutExpired(command,43200)
  with self.assertRaises(subprocess.TimeoutExpired):retry_run(['srun','timeout','43200','python','worker.py'],runner=run)
 def test_missing_timeout_and_empty_gpu_output(self):
  calls=[];sleeps=[]
  def run(command,**kw):
   calls.append(kw);return subprocess.CompletedProcess(command,0,'' if len(calls)==1 else '0, 0\n','')
  result=retry_run(['nvidia-smi','--query-gpu=index,memory.used'],runner=run,sleeper=sleeps.append)
  self.assertEqual(calls[0]['timeout'],45);self.assertEqual(sleeps,[60]);self.assertIn('0, 0',result.stdout)
 def test_non_gpu_arguments_unchanged(self):
  def run(command,**kw):self.assertTrue(kw['check']);return 'unchanged'
  self.assertEqual(retry_run(['python','worker.py'],check=True,runner=run),'unchanged')
if __name__=='__main__':unittest.main()
