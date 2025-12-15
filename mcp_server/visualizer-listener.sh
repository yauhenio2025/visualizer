#!/bin/bash
# Visualizer Notification Listener
# Subscribes to ntfy.sh topic and plays notifications when analysis jobs complete

NTFY_TOPIC="${VISUALIZER_NTFY_TOPIC:-visualizer-${USER}-$(echo $USER | md5sum | cut -c1-4)}"
NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
LOG_FILE="/tmp/visualizer_listener.log"
PID_FILE="/tmp/visualizer_listener.pid"

# Check if already running
if [[ -f "$PID_FILE" ]]; then
    old_pid=$(cat "$PID_FILE")
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "Listener already running (PID: $old_pid)"
        exit 0
    fi
fi

echo $$ > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

find_alarm_sound() {
    for sound in \
        "/usr/share/sounds/freedesktop/stereo/complete.oga" \
        "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga" \
        "/usr/share/sounds/gnome/default/alerts/drip.ogg" \
        "/usr/share/sounds/sound-icons/trumpet-12.wav" \
        "/System/Library/Sounds/Glass.aiff"; do  # macOS
        [[ -f "$sound" ]] && echo "$sound" && return
    done
}

ALARM_SOUND=$(find_alarm_sound)

play_alarm() {
    local repeats="${1:-5}"
    if command -v paplay &> /dev/null && [[ -n "$ALARM_SOUND" ]]; then
        for ((i=1; i<=repeats; i++)); do paplay "$ALARM_SOUND" 2>/dev/null; sleep 0.5; done
    elif command -v afplay &> /dev/null && [[ -n "$ALARM_SOUND" ]]; then  # macOS
        for ((i=1; i<=repeats; i++)); do afplay "$ALARM_SOUND" 2>/dev/null; sleep 0.5; done
    elif command -v spd-say &> /dev/null; then
        spd-say "$2" 2>/dev/null
    else
        for ((i=1; i<=repeats; i++)); do echo -e "\a"; sleep 0.3; done
    fi
}

send_notification() {
    local title="$1" message="$2" urgency="${3:-critical}"
    if command -v notify-send &> /dev/null; then
        notify-send -u "$urgency" -t 0 -i dialog-information "$title" "$message"
    elif command -v osascript &> /dev/null; then  # macOS
        osascript -e "display notification \"$message\" with title \"$title\""
    fi
    echo "[$(date '+%H:%M:%S')] $title: $message" >> "$LOG_FILE"
}

echo "[$(date '+%H:%M:%S')] Listener started - Topic: $NTFY_TOPIC" | tee -a "$LOG_FILE"

while true; do
    curl -s --no-buffer "${NTFY_SERVER}/${NTFY_TOPIC}/json" 2>/dev/null | while read -r line; do
        [[ -z "$line" ]] && continue

        if command -v jq &> /dev/null; then
            event_type=$(echo "$line" | jq -r '.event // empty' 2>/dev/null)
            title=$(echo "$line" | jq -r '.title // empty' 2>/dev/null)
            message=$(echo "$line" | jq -r '.message // empty' 2>/dev/null)
            tags=$(echo "$line" | jq -r '.tags // [] | join(",")' 2>/dev/null)
        else
            event_type=$(echo "$line" | grep -oP '"event"\s*:\s*"\K[^"]+' 2>/dev/null)
            title=$(echo "$line" | grep -oP '"title"\s*:\s*"\K[^"]+' 2>/dev/null)
            message=$(echo "$line" | grep -oP '"message"\s*:\s*"\K[^"]+' 2>/dev/null)
            tags=""
        fi

        [[ "$event_type" != "message" ]] && continue
        [[ -z "$message" ]] && continue

        echo "[$(date '+%H:%M:%S')] Received: $title" | tee -a "$LOG_FILE"

        if [[ "$tags" == *"completed"* ]] || [[ "$title" == *"Complete"* ]] || [[ "$title" == *"Downloaded"* ]]; then
            send_notification "🎉 $title" "$message"
            play_alarm 7 "Visualizer complete"
        elif [[ "$tags" == *"failed"* ]] || [[ "$title" == *"Failed"* ]] || [[ "$title" == *"Error"* ]]; then
            send_notification "❌ $title" "$message"
            play_alarm 5 "Visualizer failed"
        else
            send_notification "📬 $title" "$message"
            play_alarm 3 "Visualizer notification"
        fi
    done

    echo "[$(date '+%H:%M:%S')] Reconnecting..." >> "$LOG_FILE"
    sleep 5
done
