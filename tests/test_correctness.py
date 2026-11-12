"""Focused regressions supplement the repository's keyword/status CSV evaluator.

Run: python -B -m unittest discover -s tests -p test_correctness.py -v
"""
import copy
import csv
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from chatbot import Chatbot
from evaluate import label
from retriever import Retriever
from programmes import detect_in_query


class CorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = Retriever()
        cls.bot = Chatbot(cls.r)

    def test_csv_regressions(self):
        with (ROOT / 'tests/correctness_regressions.csv').open(encoding='utf-8') as f:
            for row in csv.DictReader(f):
                with self.subTest(id=row['id'], question=row['question']):
                    ans = self.bot.ask(row['question'], programme=row['programme'])
                    self.assertEqual(label(row, ans), 'Correct', ans.text)
                    self.assertEqual(ans.status, row['expected_status'])
                    if ans.status == 'answer':
                        self.assertTrue(ans.sources)

    def test_accommodation_refers_to_official_page(self):
        official = 'https://university.help.edu.my/campus-life/accommodation/'
        for q in ['What is the hostel fee?', 'Is there campus accommodation?', 'housing options']:
            with self.subTest(question=q):
                ans = self.bot.ask(q)
                self.assertIn(official, ans.text)
                self.assertNotIn('living-lifestyle', ans.text)
                self.assertTrue(any('§18.3' in s for s in ans.sources))

    def test_no_dangling_table_reference_or_unrelated_detail(self):
        # The "tables below" lead-in has no tables in the answer, so it is not evidence.
        for q, p in [('studying while working part time', 'MAI_CONV'),
                     ('When would I study Applied Programming as a part-time student?', 'MDS')]:
            with self.subTest(question=q):
                ans = self.bot.ask(q, programme=p)
                self.assertNotIn('tables below', ans.text)
                self.assertNotEqual(ans.status, 'answer')
        # A loosely related passage from another section must not ride along.
        for q in ['where can I stay on campus?', 'Where can I stay near the university?']:
            with self.subTest(question=q):
                ans = self.bot.ask(q)
                self.assertEqual(ans.status, 'answer')
                self.assertEqual(ans.details, [])
                self.assertTrue(any('§18.3' in s for s in ans.sources))
                self.assertFalse(any('Dean' in s for s in ans.sources))

    def test_working_while_studying_is_clarified(self):
        for q in ['Studying while working part time', 'Can I study while holding a job?',
                  'I work full time, can I still do the master?', 'Can I do the master while working?']:
            for p in ['MAI_CONV', None]:     # also without a selected programme
                with self.subTest(question=q, programme=p):
                    ans = self.bot.ask(q, programme=p)
                    self.assertEqual(ans.status, 'clarify')
                    self.assertIn('part-time study options', ans.text)
                    self.assertEqual(ans.sources, [])

    def test_explicit_part_time_questions_are_not_clarified(self):
        ans = self.bot.ask('How long will it take if I study while working?', programme='MAI_CONV')
        self.assertEqual(ans.status, 'answer')
        self.assertIn('2–5 years part time', ans.text)
        self.assertIn('Assuming part-time study', ans.text)
        self.assertEqual(self.bot.ask('How long will it take if I study while working?', programme=None).lead,
                         'Which programme do you mean?')
        ans = self.bot.ask('Can I study part time?', programme='MAI_CONV')
        self.assertEqual(ans.status, 'answer')
        self.assertIn('minimum 2 years, maximum 5 years', ans.text)
        self.assertTrue(any('Part Time' in s for s in ans.sources))
        self.assertEqual(self.bot.ask('Can I study part time?', programme=None).lead, 'Which programme do you mean?')
        ans = self.bot.ask('When are the part-time classes?', programme='MAI_CONV')
        self.assertEqual(ans.status, 'not_found')
        self.assertIn('does not give a class timetable', ans.text)
        self.assertNotIn('tables below', ans.text)

    def test_job_wording_alone_does_not_trigger_study_clarification(self):
        for q in ['What jobs can I get after graduation?', 'What career opportunities are there after the AI master?',
                  'Can I work while studying?']:
            with self.subTest(question=q):
                self.assertNotEqual(self.bot.ask(q, programme='MAI_CONV').intent, 'work_study')
        # Changing status is not a request about part-time availability (the held-out miss stays as it was).
        self.assertNotEqual(self.bot.ask('Can I switch from full time to part time?', programme='MDS').intent,
                            'part_time_option')

    def test_table_row_answers_still_supported(self):
        ans = self.bot.ask('Which lecturer supervises chatbot projects?')
        self.assertEqual(ans.status, 'answer')
        self.assertIn('Naline', ans.text)
        ans = self.bot.ask('Who is the dean?')
        self.assertEqual(ans.status, 'answer')
        self.assertIn('Lee Chee Leong', ans.text)

    def test_credit_values_come_from_source_rows(self):
        r = copy.copy(self.r)
        r.chunks = copy.deepcopy(self.r.chunks)
        for c in r.chunks:
            if c['section_no'] == '15.2':
                c['units'] = [u.replace('Credit Hours: 4.', 'Credit Hours: 7.')
                              if 'Subject Code: MAI503;' in u else u for u in c['units']]
        ans = Chatbot(r).ask('How many credits is MAI503?')
        self.assertIn('7 credit hours', ans.text)
        self.assertNotIn('4 credit hours', ans.text)
        self.assertTrue(any('§15.2' in s for s in ans.sources))
        self.assertTrue(any('§5.3' in s for s in ans.sources))

    def test_study_load_filters_rows(self):
        ans = self.bot.ask('How many credits is MAI503 part time?')
        self.assertIn('Part time semester table', ans.text)
        self.assertNotIn('Full time semester table', ans.text)
        self.assertFalse(any('(Full Time)' in s for s in ans.sources))

    def test_shared_module_resolves_within_selected_programme(self):
        for p in ['MAI_CONV', 'MAI_ODL', 'MCS']:
            with self.subTest(programme=p):
                ans = self.bot.ask('How many credits is MCS502?', programme=p)
                self.assertEqual(ans.status, 'answer')
                self.assertEqual(ans.programme, p)
                allowed = {c['section'] for c in self.r.chunks if p in c['programmes']}
                from chatbot import cite
                self.assertTrue(set(ans.sources) <= {cite(s) for s in allowed})

    def test_comparison_options_do_not_reask_comparison(self):
        for p in ['MAI_CONV', 'MDS']:
            ans = self.bot.ask('Compare Data Science and Cybersecurity entry requirements', programme=p)
            self.assertEqual({o['value'] for o in ans.options}, {'MDS', 'MCS'})
            for opt in ans.options:
                followup = self.bot.ask(opt['question'], programme=opt['value'])
                self.assertEqual(followup.status, 'answer')
                sec = '§7.3' if opt['value'] == 'MCS' else '§7.1'
                self.assertTrue(all(sec in s for s in followup.sources))

    def test_keep_cannot_answer_foreign_programme_question(self):
        ans = self.bot.ask('What is the minimum CGPA for cybersecurity?', keep_programme=True)
        self.assertEqual(ans.status, 'meta')
        self.assertEqual(ans.programme, 'MAI_CONV')
        self.assertFalse(ans.sources)
        self.assertNotIn('2.50', ans.text)

    def test_unsupported_facts_without_programme(self):
        for q in ['Who teaches MAI503?', 'How long does it take to get to campus from KL Sentral?',
                  'What percentage of AI graduates get a job within six months?']:
            self.assertEqual(self.bot.ask(q, programme=None).status, 'not_found')

    def test_comparison_with_for_resolves_all_named_programmes(self):
        cases = [
            ('Compare the entry requirements for Data Science and Cybersecurity', {'MDS', 'MCS'}),
            ('Compare the entry requirements for AI Conventional and Data Science', {'MAI_CONV', 'MDS'}),
            ('Compare Cybersecurity and Data Science requirements', {'MDS', 'MCS'}),
            ('Compare AI Conventional and Cybersecurity ODL entry requirements', {'MAI_CONV', 'MCS'}),
            ('Compare PhD in Computing and PhD in Computer Science entry requirements', {'PHD_COMP', 'PHD_CS'}),
        ]
        sections = {'MDS':'§7.1', 'MCS':'§7.3', 'MAI_CONV':'§7.2', 'PHD_COMP':'§7.4', 'PHD_CS':'§7.5'}
        for q, expected in cases:
            with self.subTest(question=q):
                self.assertEqual(detect_in_query(q), expected)
                for selected in ['MCS', 'MAI_CONV', 'MDS']:
                    ans = self.bot.ask(q, programme=selected)
                    self.assertEqual(ans.status, 'clarify')
                    self.assertEqual({o['value'] for o in ans.options}, expected)
                    for opt in ans.options:
                        followup = self.bot.ask(opt['question'], programme=opt['value'])
                        self.assertEqual(followup.status, 'answer')
                        self.assertEqual(followup.programme, opt['value'])
                        self.assertTrue(all(sections[opt['value']] in s for s in followup.sources))

    def test_short_semester_query_uses_exact_scoped_lookup(self):
        for p in ['MCS', 'MAI_CONV', 'MAI_ODL']:
            with self.subTest(programme=p), patch.object(self.bot, '_general', side_effect=AssertionError('Unexpected general retrieval')):
                short = self.bot.ask('semester do I take mai503', programme=p)
                long = self.bot.ask('which semester do I take mai503', programme=p)
                self.assertEqual((short.status,short.lead,short.details,short.sources,short.options),
                                 (long.status,long.lead,long.details,long.sources,long.options))
                self.assertEqual(short.programme,p)
                if p == 'MCS':
                    self.assertEqual(short.status,'not_found')
                    self.assertIn('MAI503 is not listed',short.text)
                    self.assertEqual({o['value'] for o in short.options}, {'MAI_CONV','MAI_ODL'})
                    self.assertNotIn('MDS503',short.text)
                    self.assertNotIn('MAI506',short.text)
                else:
                    self.assertEqual(short.status,'answer')
                    self.assertIn('MAI503 Natural Language Processing',short.text)
                    section = '§5.3' if p == 'MAI_CONV' else '§5.4'
                    self.assertTrue(all(section in s for s in short.sources))
                    if p == 'MAI_CONV':
                        self.assertIn('Full time: Semester 2',short.text)
                        self.assertIn('Part time: Semester 5',short.text)

    def test_other_exact_codes_and_ambiguous_codes_never_use_general(self):
        cases = [('MAI_CONV','mai506','answer'),('MCS','mcs502','answer'),
                 ('MAI_CONV','mai999','not_found'),('MAI_CONV','mai502','clarify')]
        with patch.object(self.bot,'_general',side_effect=AssertionError('Unexpected general retrieval')):
            for p,code,status in cases:
                with self.subTest(programme=p,code=code):
                    ans = self.bot.ask('semester do I take '+code,programme=p)
                    self.assertEqual(ans.status,status)
                    if status == 'answer':
                        self.assertIn(code.upper(),ans.text)
                        self.assertNotIn('MDS503',ans.text)
                    if code == 'mai506':
                        self.assertIn('Full time: Semester 3',ans.text)
                        self.assertIn('Part time: Semester 6',ans.text)
                    if code == 'mai502':
                        self.assertEqual({o['label'] for o in ans.options},
                                         {'Data Management','Deep Learning for Computer Vision'})
            for q in ['Tell me the details of MAI999', 'What about MAI999?',
                      'semester do I take mai503 and mai506']:
                self.assertIn(self.bot.ask(q).status,{'not_found','clarify'})

    def test_code_without_a_requested_field_offers_scoped_choices(self):
        with patch.object(self.bot,'_general',side_effect=AssertionError('Unexpected general retrieval')):
            ans = self.bot.ask('What about MAI503?')
            self.assertEqual(ans.status,'clarify')
            self.assertEqual({o['label'] for o in ans.options},{'Semester','Credits'})
            for opt in ans.options:
                reply = self.bot.ask(opt['value'])
                self.assertEqual(reply.status,'answer')
                self.assertIn('Natural Language Processing',reply.text)


if __name__ == '__main__':
    unittest.main()
