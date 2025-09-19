# Collection Notes AI - Taxonomy Labeling Guide

This guide provides clear instructions for categorizing customer emails according to our sentiment and intent taxonomy. Use this guide for manual labeling validation and training automated classification systems.

## Quick Reference

### Intent Categories (Choose ONE)
- **Payment Inquiry** - Customer asking about payment status/process
- **Invoice Management** - Customer reporting invoice issues or requesting changes
- **Information Request** - Customer seeking general information/clarification

### Sentiment Categories (Choose ONE)
- **Cooperative** - Customer willing to work toward resolution
- **Administrative** - System/process issues, cancellations, operational updates
- **Informational** - Routine business updates, payment confirmations, non-collection communications
- **Frustrated** - Customer expressing dissatisfaction or urgency

---

## Detailed Intent Category Guide

### 🔵 Payment Inquiry
**When to use:** Customer is asking about payment status, confirming payments, or seeking payment-related information.

**✅ INCLUDE emails that:**
- Request payment status updates ("What's the status of my payment?")
- Confirm payment has been sent ("Payment initiated via check #123")
- Ask about payment timelines ("When will this be processed?")
- Seek clarification on payment methods ("Can I pay by credit card?")
- Request case/account updates related to payments

**❌ EXCLUDE emails that:**
- Report problems with invoices (use Invoice Management instead)
- Ask for general company information (use Information Request instead)
- Only mention payment as context but focus on other issues

**Examples:**
```
✅ "Can you provide an update on case #12345 payment status?"
✅ "Payment has been initiated for invoice INV789 - please confirm receipt"
✅ "Will my payment of $500 be processed this week?"
✅ "Advise if I can pay this invoice with a credit card"

❌ "The invoice amount is incorrect - please update" (Invoice Management)
❌ "What's your company's mailing address?" (Information Request)
```

---

### 🔵 Invoice Management
**When to use:** Customer is reporting invoice problems, requesting changes, or needs invoice-related documentation.

**✅ INCLUDE emails that:**
- Request invoice corrections ("Wrong company name on invoice")
- Ask for invoice documentation ("Need your W9 form")
- Report invoice processing problems ("Can't process - missing PO number")
- Request invoice cancellations ("Please cancel duplicate invoice")
- Loop in AP teams to resolve invoice issues

**❌ EXCLUDE emails that:**
- Only ask about payment status (use Payment Inquiry instead)
- Request general company information (use Information Request instead)
- Confirm payments without mentioning invoice problems

**Examples:**
```
✅ "Please update the product name listed on invoice INV123"
✅ "We need your W9 form to process this payment"
✅ "Cancel invoice INV456 - this was billed twice"
✅ "Looping in our AP team to resolve the PO number issue"

❌ "When will my payment be processed?" (Payment Inquiry)
❌ "What's your phone number?" (Information Request)
```

---

### 🔵 Information Request
**When to use:** Customer seeks general information, procedures, or clarification not related to specific payment/invoice issues.

**✅ INCLUDE emails that:**
- Ask about company procedures ("How do we submit invoices?")
- Request general contact information
- Ask for quotes or general business information
- Seek feedback or survey responses
- Administrative/procedural questions

**❌ EXCLUDE emails that:**
- Ask about specific payment status (use Payment Inquiry instead)
- Report specific invoice problems (use Invoice Management instead)

**Examples:**
```
✅ "How do we submit invoices through your portal system?"
✅ "Please provide the contact for your accounts department"
✅ "Please find attached draft quote as requested"

❌ "What's the status of invoice INV123?" (Invoice Management)
❌ "When will my payment be processed?" (Payment Inquiry)
```

---

## Detailed Sentiment Category Guide

### 😊 Cooperative
**When to use:** Customer demonstrates willingness to resolve issues, provides helpful information, or shows positive engagement.

**✅ INCLUDE emails that:**
- Offer to provide additional information ("Happy to send the W9")
- Apologize for delays ("Sorry for the delay in responding")
- Confirm they will take action ("Will process payment tomorrow")
- Show willingness to work together ("Let me know what else you need")
- Actively try to resolve issues ("Looping in our team to fix this")

**Key phrases:** "happy to," "apologies," "will send," "let me know," "trying to resolve"

**Examples:**
```
✅ "Apologies for the delay, payment will be processed tomorrow"
✅ "Happy to provide the W9 form you requested"
✅ "Let me loop in our AP team to resolve this quickly"
✅ "Thanks for your patience while we sort this out"
```

---

### ⚙️ Administrative
**When to use:** Customer reports system/process issues, requests cancellations, or provides operational status updates.

**✅ INCLUDE emails that:**
- Report system or processing problems ("Unexpected error occurred")
- Request cancellations due to errors ("Cancel the duplicate invoice")
- Provide administrative status updates ("Currently awaiting approval")
- Mention processing delays or technical issues

**Key phrases:** "cancel the invoice," "awaiting payment," "processed and awaiting," "unexpected error," "system issue"

**Examples:**
```
✅ "Please cancel invoice #INV123 due to system error"
✅ "Invoice processed and currently awaiting final approval"
✅ "Unexpected error occurred during payment processing"
✅ "The payment is stuck in our approval system"
```

---

### 📄 Informational
**When to use:** Customer provides routine business updates, payment confirmations, or non-collection related communications.

**✅ INCLUDE emails that:**
- Confirm successful payment completion ("Invoice has been paid")
- Provide routine business updates or quotes ("Draft quote attached")
- Request feedback or procedural information ("How do we submit invoices?")
- General business communications not directly collection-related

**Key phrases:** "invoice has been paid," "please submit through," "draft quote," "opinion matters," "feedback request"

**Examples:**
```
✅ "Your invoice has been paid via check #12345"
✅ "Please submit future invoices through our online portal"
✅ "Please find attached draft quote as requested"
✅ "Your opinion matters - tell us how we're doing"
```

---

### 😠 Frustrated
**When to use:** Customer expresses dissatisfaction, urgency, or negative emotions.

**✅ INCLUDE emails that:**
- Express dissatisfaction ("This is unacceptable")
- Use urgent/demanding language ("Need immediate resolution")
- Mention escalation ("Will contact your manager")
- Show impatience ("Third time requesting this")
- Express disappointment or anger

**Key phrases:** "unacceptable," "immediately," "escalate," "disappointed," "fed up"

**Examples:**
```
✅ "This is the third time I've requested this information"
✅ "Need immediate resolution - this delay is unacceptable"
✅ "I'm escalating this to your management team"
✅ "Extremely disappointed with your service"
```

---

## Modifier Flags (Optional)

### 🚨 Urgency
Add this flag when email indicates time-sensitive matter:
- "urgent," "asap," "immediately," "deadline," "overdue"

### ⬆️ Escalation
Add this flag when customer mentions involving management:
- "manager," "supervisor," "escalate," "complaint," "legal"

### 💰 Payment Commitment
Add this flag when customer makes specific payment promise:
- "will pay," "payment scheduled," "check sent," "processing payment"

---

## Difficult Cases & Edge Cases

### When Multiple Intents Appear
**Choose the PRIMARY intent** - what is the main action the customer wants?

```
Example: "The invoice amount is wrong, and when will my payment be processed?"
→ Primary: Invoice Management (fixing the amount is the main issue)
→ Secondary: Payment Inquiry (but this comes after the fix)
```

### When Sentiment is Mixed
**Choose the DOMINANT tone** throughout the email:

```
Example: "Thanks for your help, but this delay is really frustrating"
→ Sentiment: Frustrated (despite polite opening, frustration is the main emotion)
```

### When Information is Limited
If email content is unclear or very brief:
- **Intent:** Choose Information Request (most general category)
- **Sentiment:** Choose Neutral (safest default)
- **Add note:** Flag for human review

---

## Quality Control Checklist

Before finalizing your categorization:

- [ ] **Intent:** Does this email's main purpose fit the chosen category?
- [ ] **Sentiment:** Does the overall tone match the chosen sentiment?
- [ ] **Exclusions:** Did I check what this email should NOT be categorized as?
- [ ] **Confidence:** Am I confident in this categorization? If not, flag for review.

---

## Common Mistakes to Avoid

1. **Don't confuse intent with sentiment**
   - Intent = What they want (payment info, invoice fix, general info)
   - Sentiment = How they feel (cooperative, neutral, frustrated)

2. **Don't over-analyze polite formalities**
   - "Thank you" at the end doesn't make frustrated emails cooperative

3. **Don't let subject lines override email content**
   - Focus on the email body content for categorization

4. **Don't create new categories**
   - Force-fit into existing categories or flag for taxonomy review

---

*This guide covers 100% of observed email patterns from our dataset analysis. For emails that don't fit these patterns, flag for taxonomy review and potential category expansion.*