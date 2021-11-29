import csv
import re

reply_delimiters = [
    r"^-- \n",
    r"^From: ",
    r"^On.*wrote:$"
]

class Email:
    def __init__(self, index, fields):
        self.index = index
        self.fields = fields

    def get_field(self, name):
        return self.fields[self.index[name]]

    def get_body(self):
        return self.fields[self.index["Body"]]

    def __str__(self):
        return str(self.fields)

    def get_message_with_no_reply(self):
        email_body = self.get_body()
        split = self.find_first_reply_delimiter()
        return email_body[0:split]

    def get_reply_and_signatures(self):
        email_body = self.get_body()
        split = self.find_first_reply_delimiter()
        return email_body[split:-1]

    def find_first_reply_delimiter(self):
        email_body = self.get_body()
        min = len(email_body)
        for pattern in reply_delimiters:
            m = re.search(pattern, email_body, flags=re.MULTILINE)
            if m:
                if m.start() < min:
                    min = m.start()
        return min

    def is_reply_to_other_email(self):
        return self.find_first_reply_delimiter() < len(self.get_body())

class EmailLoader:
    def __init__(self):
        self.emails = []
        self.raw_data = []
        self.column_names = []
        self.column_index = dict()
        self.email_filters = []

    def set_source_file(self, filename):
        self.filename = filename
        return self

    def set_column_names(self, names):
        self.column_names = names
        self.rebuild_column_index()

    def rebuild_column_index(self):
        reverse = lambda ab : (ab[1], ab[0])
        self.column_index = dict(map(reverse, enumerate(self.column_names)))

    def add_filter(self, func):
        self.email_filters.append(func)

    def get_filtered_emails(self):
        result = self.get_all_emails()
        for f in self.email_filters:
            result = list(filter(f,result))
        return result

    def get_all_emails(self):
        return self.emails
    
    def read_from_file(self):
        with open(self.filename, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=",")
            self.raw_data = list(reader)
            print("Loaded", len(self.raw_data), " emails.")
            self.set_column_names(self.raw_data[0])
            del self.raw_data[0]
            self.emails = list(map(lambda x : Email(self.column_index, x), self.raw_data))

loader = EmailLoader()
loader.set_source_file("./data/sent_email_export.csv")
loader.read_from_file()
loader.add_filter(lambda e : "is inviting you to a scheduled Zoom meeting." not in e.get_field("Body"))

all_emails = loader.get_all_emails()
filtered_emails = loader.get_filtered_emails()

print(len(all_emails))
print(len(filtered_emails))

b = filtered_emails[0].get_message_with_no_reply()
print(b)

b = filtered_emails[0].get_reply_and_signatures()
print(b)



