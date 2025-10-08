# CLAUDE.md - Phase 2: NetSuite User Event Script for Email Classification

This file provides guidance to Claude Code (claude.ai/code) when working on Phase 2 of the Collections Email Classification project.

## Project Context

This is **Phase 2** of a two-phase project to build an AI-powered email classification system for NetSuite Collection Notes.

**Phase 1 (Complete)**: Built a taxonomy discovery pipeline that analyzed ~700 real collection emails and produced:
- Production-ready taxonomy with 3 intent and 4 sentiment categories
- Comprehensive labeling guide with examples and decision rules
- Production `system_prompt.txt` file for LLM classification

**Phase 2 (This Project)**: Develop a NetSuite User Event Script that automatically classifies incoming collection emails using OpenAI's API and populates Collection Note records with intent and sentiment classifications.

## Business Goal

Automatically classify incoming customer emails in NetSuite Collection Notes by:
1. **Intent** - What the customer wants (Payment Inquiry, Invoice Management, Information Request)
2. **Sentiment** - How the customer feels (Cooperative, Administrative, Informational, Frustrated)

This automation will help collection agents:
- Quickly understand email purpose and tone
- Prioritize responses based on sentiment
- Track common customer inquiry patterns
- Improve response times and quality

## Technical Overview

### NetSuite User Event Script Architecture

```
Incoming Email Received (Message Record)
    ↓
User Event Script (afterSubmit)
    ↓
Filter: Only process emails attached to Open + In Dunning invoices
    ↓
Anonymize email content (PII removal)
    ↓
Retrieve system_prompt.txt from File Cabinet
    ↓
Call OpenAI API with email + system prompt
    ↓
Parse JSON response (intent, sentiment, confidence, etc.)
    ↓
Create/Update Collection Note record with classification
    ↓
Log classification metadata for audit trail
```

### Script Trigger Details

- **Record Type**: Message (incoming email)
- **Event Type**: afterSubmit
- **Execution Context**: userEventScript
- **Deployment**: Active on all incoming Message records

### Filtering Logic

The script should **only process** incoming emails that meet these criteria:

1. **Message Type**: Incoming email (not outgoing, not internal note)
2. **Related Invoice**: Email is linked to an Invoice record
3. **Invoice Status**: Invoice is "Open" (not paid, not closed)
4. **Dunning Status**: Invoice has custom field `custbody_in_dunning` = true

This prevents unnecessary API calls for non-collection emails.

## NetSuite Data Model

### Message Record (Email)

**Standard Fields**:
- `subject` - Email subject line
- `message` - Email body content
- `author` - Sender (customer)
- `recipient` - Receiver (collection agent)
- `incomingMessage` - Boolean (true for incoming)
- `transaction` - Related Invoice record ID

**Key Methods**:
```javascript
// Check if message is incoming
var isIncoming = messageRecord.getValue({fieldId: 'incomingMessage'});

// Get related invoice
var invoiceId = messageRecord.getValue({fieldId: 'transaction'});

// Get email content
var subject = messageRecord.getValue({fieldId: 'subject'});
var body = messageRecord.getValue({fieldId: 'message'});
```

### Invoice Record

**Custom Field for Dunning Status**:
- `custbody_in_dunning` - Checkbox (true = invoice is actively being dunned)

**Standard Fields**:
- `status` - Invoice status (Open, Paid Inv Full, Closed, etc.)

**Lookup from Message**:
```javascript
var invoiceStatus = search.lookupFields({
    type: search.Type.INVOICE,
    id: invoiceId,
    columns: ['status', 'custbody_in_dunning']
});
```

### Collection Note Record (Custom Record Type)

**Assumed Custom Record Type ID**: `customrecord_collection_note`

**Custom Fields**:
- `custrecord_cn_related_invoice` - List/Record (link to Invoice)
- `custrecord_cn_related_message` - List/Record (link to Message)
- `custrecord_cn_intent` - List/Dropdown (Payment Inquiry, Invoice Management, Information Request)
- `custrecord_cn_sentiment` - List/Dropdown (Cooperative, Administrative, Informational, Frustrated)
- `custrecord_cn_confidence` - Text (high, medium, low)
- `custrecord_cn_ai_reasoning` - Long Text (explanation from LLM)
- `custrecord_cn_key_phrases` - Long Text (JSON array of key phrases)
- `custrecord_cn_suggested_action` - Long Text (recommended next step)
- `custrecord_cn_classified_date` - Date (when classification occurred)
- `custrecord_cn_model_version` - Text (e.g., "gpt-4-turbo-preview")
- `custrecord_cn_anonymized_content` - Long Text (anonymized email for audit)

**Creating a Collection Note**:
```javascript
var collectionNoteRecord = record.create({
    type: 'customrecord_collection_note',
    isDynamic: true
});

collectionNoteRecord.setValue({
    fieldId: 'custrecord_cn_related_invoice',
    value: invoiceId
});

collectionNoteRecord.setValue({
    fieldId: 'custrecord_cn_related_message',
    value: messageId
});

// Set classification fields
collectionNoteRecord.setValue({
    fieldId: 'custrecord_cn_intent',
    value: intentValue // Internal ID from list
});

var noteId = collectionNoteRecord.save();
```

## Email Anonymization

### PII Patterns to Remove

The script must anonymize personally identifiable information (PII) before sending to OpenAI:

1. **Email Addresses**: `john.doe@example.com` → `[EMAIL_1]`
2. **Phone Numbers**: `555-123-4567` → `[PHONE_1]`
3. **Names**: Extract from email headers/signatures → `[NAME_1]`
4. **Company Names**: Identify domain-based companies → `[COMPANY_1]`
5. **Account Numbers**: Patterns like `ACCT-12345` → `[ACCOUNT_1]`
6. **Invoice Numbers**: Preserve structure but anonymize → `[INVOICE_1]`
7. **Addresses**: Street addresses → `[ADDRESS_1]`
8. **Tax IDs**: SSN, EIN patterns → `[TAX_ID_1]`

### Anonymization Strategy

Use regex patterns to detect and replace PII:

```javascript
function anonymizeEmail(emailText) {
    var anonymized = emailText;
    var replacementMap = {}; // Track replacements for consistency

    // Email addresses
    var emailRegex = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g;
    anonymized = anonymized.replace(emailRegex, function(match) {
        if (!replacementMap[match]) {
            var index = Object.keys(replacementMap).length + 1;
            replacementMap[match] = '[EMAIL_' + index + ']';
        }
        return replacementMap[match];
    });

    // Phone numbers (multiple formats)
    var phoneRegex = /(\+?1?[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/g;
    anonymized = anonymized.replace(phoneRegex, function(match) {
        if (!replacementMap[match]) {
            var index = Object.keys(replacementMap).length + 1;
            replacementMap[match] = '[PHONE_' + index + ']';
        }
        return replacementMap[match];
    });

    // Add more patterns as needed...

    return {
        anonymizedText: anonymized,
        replacementMap: replacementMap
    };
}
```

**Important**: The script should store the anonymized email content in the Collection Note for audit purposes, but should NOT store the original PII-containing email.

## OpenAI API Integration

### File Cabinet System Prompt

**File Location**: `/SuiteScripts/AI_Classification/system_prompt.txt`

**Retrieving the File**:
```javascript
var fileId = getSystemPromptFileId(); // Function to search for file
var fileObj = file.load({id: fileId});
var systemPrompt = fileObj.getContents();
```

**Alternative**: Store file ID in Script Parameters for faster retrieval:
```javascript
var scriptObj = runtime.getCurrentScript();
var systemPromptFileId = scriptObj.getParameter({name: 'custscript_system_prompt_file_id'});
```

### OpenAI API Call

**Endpoint**: `https://api.openai.com/v1/chat/completions`

**Request Format**:
```javascript
var payload = {
    model: 'gpt-4-turbo-preview', // Or 'gpt-4o', 'gpt-4', etc.
    messages: [
        {
            role: 'system',
            content: systemPrompt
        },
        {
            role: 'user',
            content: 'Subject: ' + anonymizedSubject + '\n\nBody: ' + anonymizedBody
        }
    ],
    response_format: {type: 'json_object'}, // Force JSON response
    temperature: 0.1, // Low temperature for consistent classification
    max_tokens: 1000
};

var response = https.post({
    url: 'https://api.openai.com/v1/chat/completions',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + getOpenAIApiKey()
    },
    body: JSON.stringify(payload)
});
```

**API Key Storage**: Store OpenAI API key in Script Parameters (encrypted):
```javascript
var scriptObj = runtime.getCurrentScript();
var apiKey = scriptObj.getParameter({name: 'custscript_openai_api_key'});
```

### Expected OpenAI Response

```json
{
    "intent": "Payment Inquiry",
    "sentiment": "Cooperative",
    "confidence": "high",
    "reasoning": "Customer is confirming payment sent and requesting receipt confirmation",
    "key_phrases": ["payment initiated", "please confirm receipt"],
    "suggested_action": "Confirm payment receipt and update case status",
    "extracted_entities": {
        "invoice_numbers": ["[INVOICE_1]"],
        "amounts": ["$1,250.00"],
        "dates": ["2024-01-15"]
    }
}
```

**Parsing Response**:
```javascript
var responseBody = JSON.parse(response.body);
var classificationResult = JSON.parse(responseBody.choices[0].message.content);

var intent = classificationResult.intent;
var sentiment = classificationResult.sentiment;
var confidence = classificationResult.confidence;
var reasoning = classificationResult.reasoning;
```

## Script Implementation Guide

### Script Structure

```javascript
/**
 * @NApiVersion 2.1
 * @NScriptType UserEventScript
 * @NModuleScope SameAccount
 */
define(['N/record', 'N/search', 'N/https', 'N/file', 'N/runtime', 'N/log'],
    function(record, search, https, file, runtime, log) {

    function afterSubmit(context) {
        try {
            // 1. Check execution context (skip on delete, copy, etc.)
            if (context.type !== context.UserEventType.CREATE &&
                context.type !== context.UserEventType.EDIT) {
                return;
            }

            // 2. Load message record
            var messageRecord = context.newRecord;

            // 3. Filter: Only process incoming emails
            if (!isIncomingEmail(messageRecord)) {
                return;
            }

            // 4. Get related invoice and check dunning status
            var invoiceId = messageRecord.getValue({fieldId: 'transaction'});
            if (!invoiceId || !isInvoiceInDunning(invoiceId)) {
                log.debug('Skip', 'Email not related to dunning invoice');
                return;
            }

            // 5. Get email content
            var subject = messageRecord.getValue({fieldId: 'subject'}) || '';
            var body = messageRecord.getValue({fieldId: 'message'}) || '';

            // 6. Anonymize email
            var anonymized = anonymizeEmail(subject + '\n\n' + body);

            // 7. Retrieve system prompt from File Cabinet
            var systemPrompt = getSystemPrompt();

            // 8. Call OpenAI API
            var classification = classifyEmail(anonymized.anonymizedText, systemPrompt);

            // 9. Create Collection Note record
            createCollectionNote(
                messageRecord.id,
                invoiceId,
                classification,
                anonymized.anonymizedText
            );

            log.audit('Success', 'Email classified and Collection Note created');

        } catch (e) {
            log.error('Error', 'Failed to classify email: ' + e.toString());
        }
    }

    function isIncomingEmail(messageRecord) {
        // Implementation
    }

    function isInvoiceInDunning(invoiceId) {
        // Implementation
    }

    function anonymizeEmail(emailText) {
        // Implementation
    }

    function getSystemPrompt() {
        // Implementation
    }

    function classifyEmail(emailText, systemPrompt) {
        // Implementation
    }

    function createCollectionNote(messageId, invoiceId, classification, anonymizedText) {
        // Implementation
    }

    return {
        afterSubmit: afterSubmit
    };
});
```

### Helper Functions

**isIncomingEmail**:
```javascript
function isIncomingEmail(messageRecord) {
    var isIncoming = messageRecord.getValue({fieldId: 'incomingMessage'});
    return isIncoming === true || isIncoming === 'T';
}
```

**isInvoiceInDunning**:
```javascript
function isInvoiceInDunning(invoiceId) {
    try {
        var invoiceFields = search.lookupFields({
            type: search.Type.INVOICE,
            id: invoiceId,
            columns: ['status', 'custbody_in_dunning']
        });

        // Check if status is "Open" and in_dunning is true
        var isOpen = invoiceFields.status &&
                     invoiceFields.status[0].value === 'open';
        var inDunning = invoiceFields.custbody_in_dunning === true;

        return isOpen && inDunning;
    } catch (e) {
        log.error('isInvoiceInDunning', 'Error checking invoice: ' + e.toString());
        return false;
    }
}
```

**getSystemPrompt**:
```javascript
function getSystemPrompt() {
    var scriptObj = runtime.getCurrentScript();
    var fileId = scriptObj.getParameter({name: 'custscript_system_prompt_file_id'});

    if (!fileId) {
        throw new Error('System prompt file ID not configured in script parameters');
    }

    try {
        var fileObj = file.load({id: fileId});
        return fileObj.getContents();
    } catch (e) {
        log.error('getSystemPrompt', 'Failed to load system prompt: ' + e.toString());
        throw e;
    }
}
```

**classifyEmail**:
```javascript
function classifyEmail(emailText, systemPrompt) {
    var scriptObj = runtime.getCurrentScript();
    var apiKey = scriptObj.getParameter({name: 'custscript_openai_api_key'});
    var model = scriptObj.getParameter({name: 'custscript_openai_model'}) || 'gpt-4-turbo-preview';

    var payload = {
        model: model,
        messages: [
            {role: 'system', content: systemPrompt},
            {role: 'user', content: emailText}
        ],
        response_format: {type: 'json_object'},
        temperature: 0.1,
        max_tokens: 1000
    };

    try {
        var response = https.post({
            url: 'https://api.openai.com/v1/chat/completions',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey
            },
            body: JSON.stringify(payload)
        });

        if (response.code !== 200) {
            throw new Error('OpenAI API returned status ' + response.code + ': ' + response.body);
        }

        var responseBody = JSON.parse(response.body);
        var classification = JSON.parse(responseBody.choices[0].message.content);

        return classification;
    } catch (e) {
        log.error('classifyEmail', 'API call failed: ' + e.toString());
        throw e;
    }
}
```

**createCollectionNote**:
```javascript
function createCollectionNote(messageId, invoiceId, classification, anonymizedText) {
    try {
        var noteRecord = record.create({
            type: 'customrecord_collection_note',
            isDynamic: true
        });

        // Link to invoice and message
        noteRecord.setValue({fieldId: 'custrecord_cn_related_invoice', value: invoiceId});
        noteRecord.setValue({fieldId: 'custrecord_cn_related_message', value: messageId});

        // Set classification results
        noteRecord.setText({fieldId: 'custrecord_cn_intent', text: classification.intent});
        noteRecord.setText({fieldId: 'custrecord_cn_sentiment', text: classification.sentiment});
        noteRecord.setValue({fieldId: 'custrecord_cn_confidence', value: classification.confidence});
        noteRecord.setValue({fieldId: 'custrecord_cn_ai_reasoning', value: classification.reasoning || ''});
        noteRecord.setValue({fieldId: 'custrecord_cn_key_phrases', value: JSON.stringify(classification.key_phrases || [])});
        noteRecord.setValue({fieldId: 'custrecord_cn_suggested_action', value: classification.suggested_action || ''});

        // Metadata
        noteRecord.setValue({fieldId: 'custrecord_cn_classified_date', value: new Date()});
        noteRecord.setValue({fieldId: 'custrecord_cn_model_version', value: runtime.getCurrentScript().getParameter({name: 'custscript_openai_model'})});
        noteRecord.setValue({fieldId: 'custrecord_cn_anonymized_content', value: anonymizedText});

        var noteId = noteRecord.save();
        log.debug('createCollectionNote', 'Collection Note created: ' + noteId);
        return noteId;
    } catch (e) {
        log.error('createCollectionNote', 'Failed to create note: ' + e.toString());
        throw e;
    }
}
```

## Script Configuration

### Script Parameters

Define these parameters in the User Event Script deployment:

| Parameter ID | Label | Type | Description |
|-------------|-------|------|-------------|
| `custscript_openai_api_key` | OpenAI API Key | Password | OpenAI API authentication key (encrypted) |
| `custscript_openai_model` | OpenAI Model | Free Text | Model name (default: gpt-4-turbo-preview) |
| `custscript_system_prompt_file_id` | System Prompt File ID | Integer | Internal ID of system_prompt.txt in File Cabinet |

### Script Deployment

1. **Script Record**: Create User Event Script record with above code
2. **Deployment**:
   - Status: Testing (initially), then Released
   - Execute As: Administrator (or dedicated API user)
   - Log Level: Debug (initially), then Error (production)
3. **Applies To**: Message record type
4. **Event Type**: After Submit
5. **Script Deployments**: All Employees (or specific roles)

## Error Handling & Logging

### Governance Limits

NetSuite User Event Scripts have governance limits:
- **API Calls**: 10 per script execution (OpenAI call counts as 1)
- **Search**: 1000 rows per script
- **Script Execution Time**: 60 seconds

**Best Practice**: Since OpenAI API calls are slow (2-5 seconds), ensure script completes within governance limits.

### Error Scenarios

1. **OpenAI API Failure**: Log error, do NOT create Collection Note
2. **Invalid JSON Response**: Log error with response body, create note with "Classification Failed" status
3. **Missing Invoice**: Log warning, skip processing
4. **File Cabinet Error**: Log error, halt script (cannot proceed without system prompt)
5. **Governance Limit**: NetSuite will automatically halt script - monitor usage

### Logging Strategy

```javascript
// Debug: Detailed execution flow (testing only)
log.debug('Step', 'Processing message ID: ' + messageId);

// Audit: Important business events
log.audit('Success', 'Email classified: ' + intent + ' / ' + sentiment);

// Error: Failures that require attention
log.error('API Error', 'OpenAI returned 429 (rate limit)');
```

## Testing Strategy

### Unit Testing (SuiteScript 2.1)

Create separate test script to validate functions:

```javascript
// Test anonymization
var testEmail = 'Contact john.doe@example.com at 555-1234';
var result = anonymizeEmail(testEmail);
log.debug('Test', result.anonymizedText); // Should show [EMAIL_1] and [PHONE_1]

// Test OpenAI API (with sample prompt)
var classification = classifyEmail('Can you confirm payment receipt?', systemPrompt);
log.debug('Test', JSON.stringify(classification));
```

### Integration Testing

1. **Test Email**: Send test email to NetSuite (from test customer account)
2. **Verify Filtering**: Ensure script only processes emails on dunning invoices
3. **Check Collection Note**: Verify all fields populated correctly
4. **Review Logs**: Check for errors or warnings
5. **Validate Anonymization**: Ensure no PII in anonymized content field

### Performance Testing

- **Monitor Governance**: Check remaining governance units in logs
- **API Response Time**: Log time taken for OpenAI API call
- **Total Execution Time**: Ensure script completes within 60 seconds

## Production Deployment Checklist

- [ ] System prompt file uploaded to File Cabinet
- [ ] OpenAI API key configured in Script Parameters
- [ ] Collection Note custom fields created
- [ ] Intent picklist values match taxonomy (Payment Inquiry, Invoice Management, Information Request)
- [ ] Sentiment picklist values match taxonomy (Cooperative, Administrative, Informational, Frustrated)
- [ ] User Event Script deployed to Message record type
- [ ] Script tested with sample emails
- [ ] Error handling validated (API failures, invalid responses)
- [ ] Governance limits confirmed (script completes within limits)
- [ ] Log level set to Error (production) or Debug (testing)
- [ ] Documentation created for collection team
- [ ] Monitoring dashboard configured (track classification volume, errors)

## Maintenance & Monitoring

### System Prompt Updates

When updating the system prompt:
1. Upload new version to File Cabinet (overwrite existing or create new version)
2. Update Script Parameter if using new file ID
3. Test with sample emails before full deployment
4. Document changes in prompt version history

### Monitoring Metrics

Track these metrics in NetSuite:
- **Classification Volume**: Number of emails classified per day/week
- **Error Rate**: Percentage of failed classifications
- **Confidence Distribution**: High vs medium vs low confidence
- **Intent Distribution**: Most common intent categories
- **Sentiment Distribution**: Most common sentiment categories

Create saved searches or dashboards to visualize these metrics.

### Common Issues & Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Script not triggering | Deployment not active | Check deployment status |
| No Collection Note created | Filtering logic excluding emails | Review filtering criteria |
| API errors (401) | Invalid API key | Verify Script Parameter |
| API errors (429) | Rate limit exceeded | Implement retry logic or reduce volume |
| Invalid JSON response | LLM not following format | Refine system prompt, add examples |
| Governance limit exceeded | Script too slow | Optimize code, reduce API timeout |

## Best Practices

### SuiteScript 2.1 Standards

- Use `@NApiVersion 2.1` for latest features
- Always include error handling with try/catch
- Log all important events (audit level)
- Use Script Parameters for configuration (never hardcode API keys)
- Validate all user inputs and API responses
- Use dynamic mode for record creation (`isDynamic: true`)

### Security Considerations

- **API Key**: Store in Script Parameters (encrypted), never in code
- **PII**: Always anonymize before sending to external API
- **Audit Trail**: Store anonymized content in Collection Note for review
- **Access Control**: Restrict script execution to appropriate roles

### Performance Optimization

- **Governance**: Monitor API call usage (only 10 allowed)
- **Caching**: Consider caching system prompt in memory (if script re-used)
- **Filtering**: Filter aggressively to avoid unnecessary API calls
- **Timeout**: Set reasonable timeout for OpenAI API (5-10 seconds)

## Support & Resources

### NetSuite Documentation
- SuiteScript 2.1 User Event: https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/section_4387799721.html
- N/https Module: https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/section_1558708810.html
- N/record Module: https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/section_4267255511.html

### OpenAI Documentation
- Chat Completions API: https://platform.openai.com/docs/api-reference/chat/create
- JSON Mode: https://platform.openai.com/docs/guides/structured-outputs

### Phase 1 Reference
- System Prompt: Located in Phase 1 repo outputs directory
- Taxonomy: Refer to Phase 1 `taxonomy.yaml`
- Labeling Guide: Use for understanding category definitions

---

*This document provides complete context for building the NetSuite User Event Script. Refer to Phase 1 deliverables for system prompt and taxonomy details.*
