import unittest
from decision_index.engines import Unsupported
from submissions.jet.engine import native_question
class ConversionTests(unittest.TestCase):
    def test_all_options_and_order_preserved(self):
        q={'type':'choice','instructions':{'task':'choose'},'criteria':{str(i):{'value':i} for i in range(255)}}
        result=native_question(q)
        self.assertEqual(list(result['criteria']),list(q['criteria']))
        self.assertEqual(len(result['criteria']),255)
        self.assertEqual(result['criteria']['254'],'{"value": 254}')
    def test_boolean_descriptions_preserved(self):
        r=native_question({'type':'noul','instructions':'Evaluate.','criteria':{'true':'Has unsupported content','false':'Fully supported'}})
        self.assertEqual(r['instructions'],'Evaluate.\nyes: Has unsupported content\nno: Fully supported')
        self.assertNotIn('criteria',r)
    def test_limits_do_not_filter(self):
        with self.assertRaises(Unsupported):native_question({'type':'choice','instructions':'Pick','criteria':{str(i):str(i) for i in range(256)}})
    def test_boolean_criteria_not_silently_dropped(self):
        with self.assertRaises(Unsupported):native_question({'type':'noul','instructions':'Pick','criteria':{'unknown':'Meaning'}})
if __name__=='__main__':unittest.main()
